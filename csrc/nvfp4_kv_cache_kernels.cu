// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// NVFP4 KV cache store kernel.
// Quantizes bf16 key/value to packed FP4 + FP8 block scales and writes them
// into the paged KV cache.
//
// Per page layout: [K_data | K_scale | V_data | V_scale]
// Both data and scale regions are contiguous per head, enabling direct
// TMA descriptor use.
//
// Reuses device functions from nvfp4_utils.cuh:
//   - cvt_warp_fp16_to_fp4()  for bf16 → fp4 quantization + block scale
//   - pack_fp4()              for packing float pairs to fp4
//   - reciprocal_approximate_ftz() for fast reciprocal

#define NVFP4_ENABLE_ELTS16 1
#include "quantization/fp4/nvfp4_utils.cuh"

#include "dispatch_utils.h"
#include "torch_utils.h"

#include <cuda_fp8.h>
#include <cmath>
#include <string>

namespace vllm {

// The cache layout is identical; this only selects the store-time scale search.
enum class NVFP4KVScaleSearch {
  DEFAULT,
  FOUR_OVER_SIX,
};

// Compute swizzled scale offset for SM100 trtllm-gen MHA kernel.
// The swizzle pattern for HND layout is:
//   [T//4, 4, 4, S//4] → permute(0, 2, 3, 1) → reshape to [T, S]
// where T = block_size (page_size), S = scale_dim = head_size // 16.
//
// For a linear (t, s) position, the swizzled position is:
//   swizzled_t = (t / 4) * 4 + (s / (S / 4))
//   swizzled_s = (s % (S / 4)) * 4 + (t % 4)
__device__ __forceinline__ int swizzle_scale_offset(int t, int s,
                                                    int scale_dim) {
  int s_group = scale_dim / 4;
  int swizzled_t = (t / 4) * 4 + (s / s_group);
  int swizzled_s = (s % s_group) * 4 + (t % 4);
  return swizzled_t * scale_dim + swizzled_s;
}

__device__ __forceinline__ float round_to_nearest_e2m1(float x) {
  const float ax = fabsf(x);
  float q;
  // Match cvt.rn.satfinite.e2m1x2.f32, including ties-to-even boundaries.
  if (ax <= 0.25f) {
    q = 0.0f;
  } else if (ax < 0.75f) {
    q = 0.5f;
  } else if (ax <= 1.25f) {
    q = 1.0f;
  } else if (ax < 1.75f) {
    q = 1.5f;
  } else if (ax <= 2.5f) {
    q = 2.0f;
  } else if (ax < 3.5f) {
    q = 3.0f;
  } else if (ax <= 5.0f) {
    q = 4.0f;
  } else {
    q = 6.0f;
  }
  return copysignf(q, x);
}

__device__ __forceinline__ void nvfp4_candidate_scale(float SFScaleVal,
                                                      float vecMax,
                                                      float denominator,
                                                      uint8_t* fp8_sf,
                                                      float* outputScale) {
  float SFValue =
      SFScaleVal * (vecMax * reciprocal_approximate_ftz(denominator));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  reinterpret_cast<__nv_fp8_e4m3&>(*fp8_sf) = tmp;
  SFValue = float(tmp);
  *outputScale = SFValue != 0.0f
                     ? reciprocal_approximate_ftz(
                           SFValue * reciprocal_approximate_ftz(SFScaleVal))
                     : 0.0f;
}

template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ float nvfp4_reconstruction_error(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float outputScale) {
  float error = 0.0f;
  const float dequantScale =
      outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    float2 vals = cast_to_float2(vec.elts[i]);
    if (outputScale == 0.0f) {
      error += vals.x * vals.x + vals.y * vals.y;
    } else {
      float qx = round_to_nearest_e2m1(vals.x * outputScale);
      float qy = round_to_nearest_e2m1(vals.y * outputScale);
      float dx = qx * dequantScale - vals.x;
      float dy = qy * dequantScale - vals.y;
      error += dx * dx + dy * dy;
    }
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    error += __shfl_xor_sync(0xffffffffu, error, 1);
  }
  return error;
}

// Blackwell-only PTX path (cvt.rn.satfinite.e2m1x2.f32): compiled only for
// SM100/SM120.  On SM70 (Volta) this instruction is unavailable, so the
// portable kernel below is used instead.
#if defined(ENABLE_NVFP4_SM100) || defined(ENABLE_NVFP4_SM120)
template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ fp4_packed_t cvt_warp_fp16_to_fp4_4over6(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float SFScaleVal, uint8_t* SFout) {
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  const float vecMax = float(__hmax(localMax.x, localMax.y));

  uint8_t sf6;
  uint8_t sf4;
  float outputScale6;
  float outputScale4;
  nvfp4_candidate_scale(SFScaleVal, vecMax, 6.0f, &sf6, &outputScale6);
  nvfp4_candidate_scale(SFScaleVal, vecMax, 4.0f, &sf4, &outputScale4);

  const float err6 =
      nvfp4_reconstruction_error<Type, CVT_FP4_NUM_THREADS_PER_SF>(
          vec, outputScale6);
  const float err4 =
      nvfp4_reconstruction_error<Type, CVT_FP4_NUM_THREADS_PER_SF>(
          vec, outputScale4);
  const bool use4 = err4 < err6;
  const float outputScale = use4 ? outputScale4 : outputScale6;

  if (SFout) *SFout = use4 ? sf4 : sf6;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = cast_to_float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  return pack_fp4(fp2Vals);
}

// Kernel: quantize bf16 key/value to NVFP4 and store in paged KV cache.
//
// Takes separate data and scale cache pointers for K and V.
// Within each KV side, data and scale are separate contiguous regions.
//
// Threading: one CUDA block per token, threads process heads and
// groups of 16 elements within each head.
template <typename scalar_t, NVFP4KVScaleSearch SCALE_SEARCH>
__global__ void reshape_and_cache_nvfp4_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_data_cache,  // data region for K
    uint8_t* __restrict__ value_data_cache,    // data region for V
    uint8_t* __restrict__ key_scale_cache,     // scale region for K
    uint8_t* __restrict__ value_scale_cache,   // scale region for V
    const int64_t* __restrict__ slot_mapping,  // [num_actual_tokens]
    const float* __restrict__ k_scale_ptr,     // pointer to checkpoint k_scale
    const float* __restrict__ v_scale_ptr,     // pointer to checkpoint v_scale
    const int64_t key_stride,                  // key.stride(0) in elements
    const int64_t value_stride,                // value.stride(0) in elements
    const int num_heads, const int head_size, const int block_size,
    const int64_t data_block_stride,         // data cache stride for dim 0
    const int64_t data_head_stride,          // data cache stride for heads
    const int64_t data_block_offset_stride,  // data cache stride for tokens
    const int64_t scale_block_stride,        // scale cache stride for dim 0
    const int64_t scale_head_stride,         // scale cache stride for heads
    const int64_t scale_block_offset_stride  // scale cache stride for tokens
) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int ELTS = CVT_FP4_ELTS_PER_THREAD;  // 16 or 8
  static constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / ELTS;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int block_offset = static_cast<int>(slot_idx % block_size);

  const int scale_dim = head_size / 16;
  const int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;

  const int total_groups = num_heads * groups_per_head;
  const int tid = threadIdx.x;
  const int num_thread_groups = blockDim.x / THREADS_PER_SF;
  const int tg_id = tid / THREADS_PER_SF;
  const int tg_lane = tid % THREADS_PER_SF;

  // Process both K (kv=0) and V (kv=1)
#pragma unroll
  for (int kv = 0; kv < 2; kv++) {
    const scalar_t* __restrict__ src = (kv == 0) ? key : value;
    const float global_scale = 1.0f / ((kv == 0) ? *k_scale_ptr : *v_scale_ptr);
    const int64_t src_stride = (kv == 0) ? key_stride : value_stride;
    uint8_t* __restrict__ data_cache =
        (kv == 0) ? key_data_cache : value_data_cache;
    uint8_t* __restrict__ sc_cache =
        (kv == 0) ? key_scale_cache : value_scale_cache;

    // Source pointer for this token (use actual stride, not assumed contiguous)
    const CudaType* __restrict__ token_src =
        reinterpret_cast<const CudaType*>(src) + token_idx * src_stride;

    // Destination bases in data and scale caches for this token's block
    uint8_t* __restrict__ data_block =
        data_cache + block_idx * data_block_stride;
    uint8_t* __restrict__ scale_block =
        sc_cache + block_idx * scale_block_stride;

    for (int g = tg_id; g < total_groups; g += num_thread_groups) {
      const int head = g / groups_per_head;
      const int group_in_head = g % groups_per_head;

      // Load 16 (or 8) bf16 elements from source
      PVec in_vec;
      const CudaType* __restrict__ src_ptr =
          token_src + head * head_size + group_in_head * CVT_FP4_SF_VEC_SIZE +
          tg_lane * ELTS;

#pragma unroll
      for (int i = 0; i < ELTS / 2; i++) {
        in_vec.elts[i] = reinterpret_cast<
            const typename PackedTypeConverter<CudaType>::Type*>(src_ptr)[i];
      }

      // Quantize: produces packed fp4 and writes scale factor.
      uint8_t sf_val;
      uint8_t* sf_out_ptr = (tg_lane == 0) ? &sf_val : nullptr;

      fp4_packed_t packed;
      if constexpr (SCALE_SEARCH == NVFP4KVScaleSearch::FOUR_OVER_SIX) {
        packed = cvt_warp_fp16_to_fp4_4over6<CudaType, THREADS_PER_SF>(
            in_vec, global_scale, sf_out_ptr);
      } else {
        packed = cvt_warp_fp16_to_fp4<CudaType, THREADS_PER_SF>(
            in_vec, global_scale, sf_out_ptr);
      }

      // Write packed FP4 data to data cache
      uint8_t* __restrict__ data_dst = data_block + head * data_head_stride +
                                       block_offset * data_block_offset_stride;

#if CVT_FP4_PACK16
      {
        // 16 elements → 8 bytes (u32x2)
        int data_byte_offset = group_in_head * 8;
        reinterpret_cast<uint64_t*>(data_dst + data_byte_offset)[0] =
            (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
      }
#else
      {
        // 8 elements → 4 bytes (uint32_t)
        int data_byte_offset =
            group_in_head * CVT_FP4_SF_VEC_SIZE / 2 + tg_lane * ELTS / 2;
        reinterpret_cast<uint32_t*>(data_dst + data_byte_offset)[0] = packed;
      }
#endif

      // Write block scale to scale cache.
      // K (kv==0): linear layout (no swizzle).
      // V (kv==1): swizzled layout for SM100 trtllm-gen MHA kernel.
      if (sf_out_ptr != nullptr) {
        int scale_idx = group_in_head;
        uint8_t* __restrict__ scale_dst;
        if (kv == 0) {
          scale_dst = scale_block + head * scale_head_stride +
                      block_offset * scale_block_offset_stride + scale_idx;
        } else {
          int swizzled_offset =
              swizzle_scale_offset(block_offset, scale_idx, scale_dim);
          int swizzled_t = swizzled_offset / scale_dim;
          int swizzled_s = swizzled_offset % scale_dim;
          scale_dst = scale_block + head * scale_head_stride +
                      swizzled_t * scale_block_offset_stride + swizzled_s;
        }
        *scale_dst = sf_val;
      }
    }
  }
}
#endif  // defined(ENABLE_NVFP4_SM100) || defined(ENABLE_NVFP4_SM120)

#if defined(ENABLE_NVFP4_SM70)

// Portable SM70 (Volta) store path: the Blackwell cvt.rn.satfinite.e2m1x2.f32
// PTX is unavailable on sm_70, so the FP4 (E2M1) rounding and FP8-E4M3 block
// scale encode are done in portable C++.

// Map an already-E2M1-rounded value to its 4-bit code.  Thresholds are the
// midpoints of the representable magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
__device__ __forceinline__ uint8_t nvfp4_sm70_e2m1_code(float v) {
  const bool neg = v < 0.0f;
  const float a = fabsf(v);
  uint8_t mag;
  if (a < 0.25f) {
    mag = 0;
  } else if (a < 0.75f) {
    mag = 1;
  } else if (a < 1.25f) {
    mag = 2;
  } else if (a < 1.75f) {
    mag = 3;
  } else if (a < 2.5f) {
    mag = 4;
  } else if (a < 3.5f) {
    mag = 5;
  } else if (a < 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return static_cast<uint8_t>(neg ? (0x8u | mag) : mag);
}

// SM70 NVFP4 store kernel: one block per token; threads grid-stride over
// (head, group) pairs, processing K then V.  Writes packed FP4 data + linear
// (or, only if requested, swizzled) FP8 block scales.
template <typename scalar_t>
__global__ void reshape_and_cache_nvfp4_kernel_sm70(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_data_cache,
    uint8_t* __restrict__ value_data_cache,
    uint8_t* __restrict__ key_scale_cache,
    uint8_t* __restrict__ value_scale_cache,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr,
    const int64_t key_stride,
    const int64_t value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int64_t data_block_stride,
    const int64_t data_head_stride,
    const int64_t data_block_offset_stride,
    const int64_t scale_block_stride,
    const int64_t scale_head_stride,
    const int64_t scale_block_offset_stride,
    const bool swizzle_v_scale) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;

  const int64_t token_idx = blockIdx.x;
  // One CTA per (token, side). At decode the token count is 1, so without this
  // the whole layer's KV write is a single block on a single SM running K and
  // V back to back - which is why 512 elements took 24.3 us.
  const int kv = static_cast<int>(blockIdx.y);
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;
  const int64_t block_idx = slot_idx / block_size;
  const int block_offset = static_cast<int>(slot_idx % block_size);

  const int scale_dim = head_size / 16;
  const int groups_per_head = head_size / 16;
  const int total_groups = num_heads * groups_per_head;

  // Four lanes per 16-value group rather than one. The per-group work is a
  // serial e2m1 encode of 16 values, and one decode token is only 32 groups, so
  // with one thread per group the whole layer's KV write sat behind a 16-deep
  // dependent chain (13.1 us for 1 KB). The block scale needs only max(|v|),
  // which is order-independent, and each value encodes on its own, so spreading
  // a group over four lanes is bit-exact.
  constexpr int kLanesPerGroup = 4;
  const int quad = static_cast<int>(threadIdx.x) / kLanesPerGroup;
  const int lane_in_group = static_cast<int>(threadIdx.x) % kLanesPerGroup;
  const int quads = static_cast<int>(blockDim.x) / kLanesPerGroup;
  // A fixed iteration count for every thread in the block: the intra-group
  // shuffles below are warp collectives, so lanes whose group is out of range
  // must still walk the loop (with a clamped, in-range index) instead of
  // exiting early.
  const int group_iters = (total_groups + quads - 1) / quads;
  for (int it = 0; it < group_iters; ++it) {
    const int g_raw = quad + it * quads;
    const bool active = (g_raw < total_groups);
    const int g = active ? g_raw : total_groups - 1;
    const int head = g / groups_per_head;
    const int group_in_head = g % groups_per_head;

    {
      const scalar_t* __restrict__ src = (kv == 0) ? key : value;
      const float global_scale =
          1.0f / ((kv == 0) ? *k_scale_ptr : *v_scale_ptr);
      const int64_t src_stride = (kv == 0) ? key_stride : value_stride;
      uint8_t* __restrict__ data_cache =
          (kv == 0) ? key_data_cache : value_data_cache;
      uint8_t* __restrict__ sc_cache =
          (kv == 0) ? key_scale_cache : value_scale_cache;

      const CudaType* __restrict__ token_src =
          reinterpret_cast<const CudaType*>(src) + token_idx * src_stride;
      const int base = head * head_size + group_in_head * 16;

      float vals[4];
      float vecMax = 0.0f;
      // This lane's slice of the group: four halves in one 8-byte load. `base`
      // is a multiple of 16 halves (32 B) and the row start is
      // allocator-aligned, so the vector load is legal.
      {
        const uint2 word = *reinterpret_cast<const uint2*>(
            token_src + base + lane_in_group * kLanesPerGroup);
        const __half2* h2 = reinterpret_cast<const __half2*>(&word);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 f2 = __half22float2(h2[h]);
          vals[2 * h] = f2.x;
          vals[2 * h + 1] = f2.y;
          vecMax = fmaxf(vecMax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        }
      }
      // Max over the group; max is order-independent, so this reproduces the
      // single-thread version's vecMax exactly. width == kLanesPerGroup keeps
      // the exchange inside one group's lanes.
#pragma unroll
      for (int m = kLanesPerGroup / 2; m > 0; m >>= 1) {
        vecMax = fmaxf(vecMax,
                       __shfl_xor_sync(0xffffffffu, vecMax, m, kLanesPerGroup));
      }

      // Block scale: SFValue = global_scale * vecMax / 6, stored as FP8-E4M3.
      const float SFValue = global_scale * (vecMax * (1.0f / 6.0f));
      __nv_fp8_e4m3 sf_tmp = __nv_fp8_e4m3(SFValue);
      const uint8_t sf_code = sf_tmp.__x;
      const float SFValue_r = float(sf_tmp);
      const float outputScale = (SFValue_r != 0.0f)
                                    ? (1.0f / (SFValue_r * (1.0f / global_scale)))
                                    : 0.0f;

      // Two packed code bytes per lane (four nibble pairs).
      const float lo0 = round_to_nearest_e2m1(vals[0] * outputScale);
      const float hi0 = round_to_nearest_e2m1(vals[1] * outputScale);
      const float lo1 = round_to_nearest_e2m1(vals[2] * outputScale);
      const float hi1 = round_to_nearest_e2m1(vals[3] * outputScale);
      const uint16_t codes2 =
          static_cast<uint16_t>((nvfp4_sm70_e2m1_code(hi0) << 4) |
                                nvfp4_sm70_e2m1_code(lo0)) |
          static_cast<uint16_t>(((nvfp4_sm70_e2m1_code(hi1) << 4) |
                                 nvfp4_sm70_e2m1_code(lo1))
                                << 8);

      uint8_t* __restrict__ data_dst =
          data_cache + block_idx * data_block_stride +
          head * data_head_stride + block_offset * data_block_offset_stride +
          group_in_head * 8;
      if (active) {
        // The group base is 8 B aligned, so lane k lands on a 2 B boundary.
        *reinterpret_cast<uint16_t*>(data_dst + lane_in_group * 2) = codes2;
      }

      uint8_t* __restrict__ scale_block = sc_cache + block_idx * scale_block_stride;
      if (active && lane_in_group == 0) {
        uint8_t* __restrict__ scale_dst;
        if (kv == 0 || !swizzle_v_scale) {
          scale_dst = scale_block + head * scale_head_stride +
                      block_offset * scale_block_offset_stride + group_in_head;
        } else {
          const int swizzled_offset =
              swizzle_scale_offset(block_offset, group_in_head, scale_dim);
          const int swizzled_t = swizzled_offset / scale_dim;
          const int swizzled_s = swizzled_offset % scale_dim;
          scale_dst = scale_block + head * scale_head_stride +
                      swizzled_t * scale_block_offset_stride + swizzled_s;
        }
        *scale_dst = sf_code;
      }
    }
  }
}
#endif  // defined(ENABLE_NVFP4_SM70)

}  // namespace vllm

// Non-template entry point callable from cache_kernels.cu.
// Receives key_cache/value_cache as kv_cache[:, 0] and kv_cache[:, 1].
// Each KV side contains both data and scale:
//   page = [K_data | K_scale | V_data | V_scale]
void reshape_and_cache_nvfp4_dispatch(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const std::string& kv_cache_dtype) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int data_dim = head_size / 2;
  int scale_dim = head_size / 16;
  int full_dim = data_dim + scale_dim;

  // key_cache is kv_cache[:, 0] with shape
  // [num_blocks, block_size, num_heads, full_dim] in logical order.
  // Strides encode the physical layout (HND or NHD).
  STD_TORCH_CHECK(key_cache.dim() == 4, "key_cache must be 4D");
  STD_TORCH_CHECK(key_cache.size(3) == full_dim,
                  "key_cache last dim must be data_dim + scale_dim, got ",
                  key_cache.size(3), " expected ", full_dim);

  int block_size = key_cache.size(1);

  STD_TORCH_CHECK(head_size % 16 == 0,
                  "head_size must be divisible by 16 for NVFP4 KV cache");
  STD_TORCH_CHECK(block_size % 4 == 0,
                  "block_size must be divisible by 4 for NVFP4 KV cache "
                  "swizzle");

  // Detect physical layout from strides (based on full_dim).
  // HND: head stride > block_offset stride.
  bool is_hnd = key_cache.stride(2) > key_cache.stride(1);

  // Cell-interleaved layout (SM70 in-service pool): each (head, token) cell
  // is a contiguous [data | scale] full_dim bytes; the token stride spans a
  // wider per-token region (K+V cells), so scale follows data inside each
  // cell instead of a per-side scale region. Distinguished from the dense
  // NHD/HND layouts by a non-dense block stride.
  const bool cell_layout =
      key_cache.stride(3) == 1 &&
      key_cache.stride(2) == (int64_t)full_dim &&
      key_cache.stride(1) % (int64_t)full_dim == 0 &&
      key_cache.stride(1) > (int64_t)full_dim &&
      key_cache.stride(0) !=
          key_cache.stride(1) * key_cache.stride(2) * key_cache.stride(3);

  int64_t data_block_stride = key_cache.stride(0);  // page_bytes
  int64_t data_head_stride, data_block_offset_stride;
  uint8_t* key_scale_ptr;
  uint8_t* value_scale_ptr;
  int64_t scale_block_stride, scale_head_stride, scale_block_offset_stride;
  if (cell_layout) {
    // Data and scale share the view's real strides; the scale region is a
    // data_dim-byte offset into the same cells.
    data_head_stride = key_cache.stride(2);
    data_block_offset_stride = key_cache.stride(1);
    key_scale_ptr = key_cache.mutable_data_ptr<uint8_t>() + data_dim;
    value_scale_ptr = value_cache.mutable_data_ptr<uint8_t>() + data_dim;
    scale_block_stride = key_cache.stride(0);
    scale_head_stride = key_cache.stride(2);
    scale_block_offset_stride = key_cache.stride(1);
  } else {
    if (is_hnd) {
      data_head_stride = (int64_t)block_size * data_dim;
      data_block_offset_stride = data_dim;
    } else {
      data_head_stride = data_dim;
      data_block_offset_stride = (int64_t)num_heads * data_dim;
    }

    // Page layout: [K_data | K_scale | V_data | V_scale]
    // Scale follows data within each KV side.
    int64_t data_per_kv = (int64_t)num_heads * block_size * data_dim;

    key_scale_ptr = key_cache.mutable_data_ptr<uint8_t>() + data_per_kv;
    value_scale_ptr = value_cache.mutable_data_ptr<uint8_t>() + data_per_kv;

    // Scale strides: same page stride, inner strides from layout.
    scale_block_stride = data_block_stride;
    if (is_hnd) {
      scale_head_stride = (int64_t)block_size * scale_dim;
      scale_block_offset_stride = scale_dim;
    } else {
      scale_head_stride = scale_dim;
      scale_block_offset_stride = (int64_t)num_heads * scale_dim;
    }
  }

  const float* k_scale_ptr = k_scale.const_data_ptr<float>();
  const float* v_scale_ptr = v_scale.const_data_ptr<float>();

  int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;
  int total_groups = num_heads * groups_per_head;
  constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  int num_threads = std::min(total_groups * THREADS_PER_SF, 512);
  num_threads = ((num_threads + 31) / 32) * 32;

#if defined(ENABLE_NVFP4_SM70)
  // The SM70 kernel hands each 16-value group to four lanes (see the kernel), so
  // it needs 4x the threads the one-thread-per-group mapping sized for.
  num_threads = std::min(total_groups * 4, 512);
  num_threads = ((num_threads + 31) / 32) * 32;
  // The SM70 kernel reads each 16-value group with four 8-byte loads and writes
  // its 8 packed code bytes with one 8-byte store, so every base it derives has
  // to stay 8-byte aligned: head_size % 16 == 0 makes the in-row group offsets
  // multiples of 32 B, but the row stride, the caller's pointer (key/value can
  // be a qkv split) and the cache strides can all be odd.  Misalignment would
  // corrupt KV silently or fault the device, so fail at startup instead.
  STD_TORCH_CHECK((reinterpret_cast<uintptr_t>(key.const_data_ptr()) & 7u) == 0 &&
                      (reinterpret_cast<uintptr_t>(
                          value.const_data_ptr()) & 7u) == 0 &&
                      key.stride(0) % 4 == 0 && value.stride(0) % 4 == 0 &&
                      (reinterpret_cast<uintptr_t>(
                          key_cache.const_data_ptr()) & 7u) == 0 &&
                      (reinterpret_cast<uintptr_t>(
                          value_cache.const_data_ptr()) & 7u) == 0 &&
                      data_block_stride % 8 == 0 && data_head_stride % 8 == 0 &&
                      data_block_offset_stride % 8 == 0,
                  "SM70 NVFP4 KV cache: misaligned KV pointer or cache stride "
                  "for the vectorised store (head_size=",
                  head_size, ", num_heads=", num_heads, ")");
  // The SM70 kernel takes its K/V side from blockIdx.y.
  dim3 grid(num_tokens, 2);
#else
  dim3 grid(num_tokens);
#endif
  dim3 block(num_threads);

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

#if defined(ENABLE_NVFP4_SM70)
  // The SM70 (Volta) linear-attention reader needs a linear V-scale block;
  // the Blackwell readers keep the upstream swizzled layout in their own
  // kernel path, so this value is only consumed by the SM70 kernel.
  int sm_major = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor,
                         key.get_device_index());
  const bool swizzle_v_scale = (sm_major != 7);
#endif

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      key.scalar_type(), "reshape_and_cache_nvfp4", [&] {
#if defined(ENABLE_NVFP4_SM70)
        // SM70 (Volta): portable path (no Blackwell cvt.e2m1x2 PTX).  The
        // 4-over-6 scale search is an SM100/SM120 refinement; both NVFP4
        // dtypes store the same layout here, so both route to this kernel.
        vllm::reshape_and_cache_nvfp4_kernel_sm70<scalar_t>
            <<<grid, block, 0, stream>>>(
                key.const_data_ptr<scalar_t>(),
                value.const_data_ptr<scalar_t>(),
                key_cache.mutable_data_ptr<uint8_t>(),
                value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                num_heads, head_size, block_size, data_block_stride,
                data_head_stride, data_block_offset_stride,
                scale_block_stride, scale_head_stride,
                scale_block_offset_stride, swizzle_v_scale);
#elif defined(ENABLE_NVFP4_SM100) || defined(ENABLE_NVFP4_SM120)
        if (kv_cache_dtype == "nvfp4") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::DEFAULT>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else if (kv_cache_dtype == "nvfp4_4over6") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::FOUR_OVER_SIX>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else {
          STD_TORCH_CHECK(false,
                          "Unsupported NVFP4 KV cache dtype: ", kv_cache_dtype);
        }
#endif
      });
}
