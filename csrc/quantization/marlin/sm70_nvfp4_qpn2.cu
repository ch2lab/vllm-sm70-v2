// SM70 NVFP4 QPN2 decode GEMM — high-occupancy decode kernel + stable-ABI ops.
//
// Ported from 1Cat-vLLM csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu
// (Apache-2.0; the execution layout derives from dnv2003/v100-skinny, MIT —
// see LICENSE.v100-skinny in that directory) for the `_C_stable_libtorch`
// (stable-ABI) build on SM70 (V100).
//
// Why: the SM70 Marlin NVFP4 GEMM is latency-bound at 25% occupancy (114
// registers and 20.48 KB of double-buffered A+B smem each cap it at 4
// blocks/SM), reaching only ~224 GB/s (36% of V100 DRAM peak) for decode
// (m<=8). This kernel feeds the WMMA m8n8k4 B-operand directly from in-register
// dequant (no smem B staging), reads A straight from gmem, and keeps shared
// memory minimal, reaching ~588 GB/s (65% peak) — ~2.6x faster. Best config:
// SplitK=8, NAcc=1.
//
// A one-time prepack (row-major [n,k/2] NVFP4 + [n,k/16] E4M3 scales -> QPN2
// column-major codes/scales) runs at weight-load time.
//
// Registers ops:
//   sm70_nvfp4_qpn2_prepare(weight, weight_scale) -> (codes, scales)
//   sm70_nvfp4_qpn2_gemm(out, input, codes, scales, global_scale, split_k, nacc)

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unistd.h>

namespace qpn2 {

constexpr int kPrepareThreads = 256;
constexpr int kQpn2RowsPerCta = 8;

__device__ __forceinline__ int qpn2_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__device__ __forceinline__ int qpn2_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

__global__ void nvfp4_qpn2_prepack_codes_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ weight, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 2;
  if (index >= numel) {
    return;
  }

  const int physical_byte = static_cast<int>(index & 7);
  size_t outer = index >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  const int logical_k0 = group * 16 + qpn2_logical_k(physical_byte * 2);
  const int logical_k1 = group * 16 + qpn2_logical_k(physical_byte * 2 + 1);
  const int k_bytes = k >> 1;
  const uint8_t packed0 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k0 >> 1)];
  const uint8_t packed1 =
      weight[static_cast<size_t>(column) * k_bytes + (logical_k1 >> 1)];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((logical_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((logical_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

__global__ void nvfp4_qpn2_prepack_scales_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ scales, int n,
    int k) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(n) * k / 16;
  if (index >= numel) {
    return;
  }

  const int lane = static_cast<int>(index & 31);
  size_t outer = index >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  output[index] = scales[static_cast<size_t>(column) * groups_k16 + group];
}

__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

// Effective group scale: the exact E4M3 value times the FP32 global scale,
// rounded to FP16 ONCE. Folding the global scale (and the 2^14 FP4 bias
// undo) into this factor before rounding loses subnormal group scales and
// overflows large ones, which silently changes the model weights.
__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                      float global_scale) {
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    // Undo the FP4 exponent bias on the code (an exact power-of-two multiply)
    // rather than on the scale, so the scale keeps a single rounding.
    const half2 code =
        __hmul2(*reinterpret_cast<half2*>(&values[index]),
                __float2half2_rn(16384.0f));
    output[index] = __hmul2(code, scale);
  }
}

#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, int RowTiles = 1>
__global__ void nvfp4_qpn2_sm70_kernel(const uint8_t* __restrict__ codes,
                                       const uint8_t* __restrict__ group_scales,
                                       const half* __restrict__ input,
                                       half* __restrict__ output,
                                       float* __restrict__ partials_out,
                                       int n, int k, int m, float global_scale,
                                       int ksplit) {
  static_assert(RowTiles == 1 || RowTiles == 2,
                "NVFP4 QPN2 supports one or two 8-row tiles");
  __shared__ float partials[SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpn2RowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_ksplit = groups_k16 / ksplit;
  const int groups_per_warp = groups_per_ksplit / SplitK;
  const int group_begin =
      blockIdx.z * groups_per_ksplit + warp * groups_per_warp;
  const uint2* code_ptr = reinterpret_cast<const uint2*>(codes) +
                          static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const uint8_t* scale_ptr =
      group_scales + static_cast<size_t>(tile) * groups_k16 * 32 + lane;

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    const uint2 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    const half2 scale = nvfp4_effective_scale(
        __ldg(scale_ptr + static_cast<size_t>(group) * 32), global_scale);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpn2RowsPerCta + local_row;
      if (row < m) {
        const half* input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    const int output_row = row_base + (element >> 5);
    const int output_col = element & 31;
    if (output_row < m) {
      const size_t out_idx =
          static_cast<size_t>(output_row) * n +
          (static_cast<size_t>(tile) * 32 + output_col);
      if (ksplit == 1) {
        output[out_idx] = __float2half(value);
      } else {
        partials_out[static_cast<size_t>(blockIdx.z) * m * n + out_idx] =
            value;
      }
    }
  }
}
// Reduce the per-ksplit float partials (layout [ksplit, m, n]) into the
// half output. Launched on the same stream right after the split kernel.
__global__ void nvfp4_qpn2_reduce_kernel(const float* __restrict__ partials,
                                         half* __restrict__ output, int m,
                                         int n, int ksplit) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(m) * n) {
    return;
  }
  const size_t row = idx / n;
  const size_t col = idx - row * n;
  float value = 0.0f;
  for (int z = 0; z < ksplit; ++z) {
    value += partials[(static_cast<size_t>(z) * m + row) * n + col];
  }
  output[idx] = __float2half(value);
}

template <int SplitK, int NAcc, int RowTiles = 1>
void launch_qpn2(const uint8_t* codes, const uint8_t* scales, const half* input,
                 half* output, float* partials_out, int n, int k, int m,
                 float global_scale, int ksplit, cudaStream_t stream) {
  constexpr int kRowsPerCta = kQpn2RowsPerCta * RowTiles;
  const dim3 grid(n / 32, (m + kRowsPerCta - 1) / kRowsPerCta, ksplit);
  nvfp4_qpn2_sm70_kernel<SplitK, NAcc, RowTiles>
      <<<grid, (32 * SplitK), 0, stream>>>(codes, scales, input, output,
                                            partials_out, n, k, m, global_scale,
                                            ksplit);
  if (ksplit > 1) {
    const int total = m * n;
    nvfp4_qpn2_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        partials_out, output, m, n, ksplit);
  }
}
// ---------------------------------------------------------------------------
// Route-tiled (grouped) QPN2 expert GEMM: the same lane-major in-register
// dequant feed as the dense kernel above, but a CTA is
//     (32 output columns) x (one moe_align_block_size route block of 16 slots)
// instead of (32 columns) x (8 contiguous rows).
//
// Why this exists: at decode batch 1 with top-8 of 256 experts, one MoE gemm
// is 8 tiny GEMMS of shape M=1, N=512|2048, K=2048|256. Tile size is
// irrelevant to how much weight has to be streamed, so the only thing that
// matters is CTA count: N/32 x route_blocks = 16 x 8 = 128 CTAs versus the 16
// a single-expert launch would produce, which is what keeps the SMs fed. The
// 16-slot route block wastes nothing but tensor-core rows (Volta's mma is
// m8n8k4, so a 16-row tile is 2 m-tiles either way); padded slots carry
// sorted_id == -1 and are skipped on both the load and the store side.
//
// K-splitting across CTAs is available (grid.z) and stays deterministic: each
// slice writes its own fp32 partial row and a reduce kernel sums them in a
// fixed order. That is deliberately not the fp16-atomicAdd split-K the SM70
// Marlin MoE kernel uses, which is not reproducible run to run.
//
// Semantics mirror the Marlin MoE gather iterator exactly: the A row of slot s
// is sorted_token_ids[s] / top_k, and the output row is sorted_token_ids[s]
// itself (the expanded-token layout the surrounding moe_align / act / sum
// kernels already use), so this is a drop-in for both gemms of a MoE layer.
// Fused gate/up activation for the w2 GEMM's A-fragment (VLLM_SM70_MOE_FUSE_ACT).
// Bit-exact with act_and_mul: silu(x) = x / (1 + exp(-x)) in fp32, rounded to
// fp16 (the packed kernel returns a half2 before the multiply), then multiplied
// by the up half in fp32 and rounded once more.
__device__ __forceinline__ half qpn2_silu_mul(half gate, half up) {
  const float g = __half2float(gate);
  const float s = g / (1.0f + expf(-g));
  const half sh = __float2half(s);
  return __float2half(__half2float(sh) * __half2float(up));
}

__device__ __forceinline__ uint4 qpn2_act_pair(const uint4& gate,
                                               const uint4& up) {
  const half* gh = reinterpret_cast<const half*>(&gate);
  const half* uh = reinterpret_cast<const half*>(&up);
  half out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = qpn2_silu_mul(gh[i], uh[i]);
  }
  uint4 r;
  memcpy(&r, out, sizeof(r));
  return r;
}

template <int SplitK, int NAcc, int RowTiles, bool ACT_INPUT = false>
__global__ void nvfp4_qpn2_moe_sm70_kernel(
    const uint8_t* __restrict__ codes,   // [E, n, k/2] QPN2-packed
    const uint8_t* __restrict__ group_scales,  // [E, n, k/16]
    const half* __restrict__ input,      // [m, k]
    half* __restrict__ output,           // [m*top_k, n]
    float* __restrict__ partials_out,    // [ksplit, m*top_k, n] or null
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_past_padded,
    const float* __restrict__ topk_weights, const float* __restrict__ global_scale,
    int n, int k, int m, int top_k, bool mul_topk_weights, int ksplit,
    int gs_shards, int tiles_per_shard, int act_stride = 0,
    int act_half = 0) {
  constexpr int kMoeBlockM = RowTiles * kQpn2RowsPerCta;
  __shared__ float partials[SplitK][RowTiles * 256];
  __shared__ int s_expanded[kMoeBlockM];
  __shared__ int s_token[kMoeBlockM];
  __shared__ float s_weight[kMoeBlockM];

  const int moe_block = blockIdx.y;
  const int padded_tokens = num_tokens_past_padded[0];
  if (moe_block * kMoeBlockM >= padded_tokens) {
    return;
  }
  const int expert = expert_ids[moe_block];
  if (expert < 0) {
    return;
  }
  const int expanded_count = m * top_k;

  if (threadIdx.x < kMoeBlockM) {
    const int route_row = moe_block * kMoeBlockM + static_cast<int>(threadIdx.x);
    int sid = -1;
    if (route_row < padded_tokens) {
      const int cand = sorted_token_ids[route_row];
      if (cand >= 0 && cand < expanded_count) {
        sid = cand;
      }
    }
    s_expanded[threadIdx.x] = sid;
    s_token[threadIdx.x] = sid >= 0 ? sid / top_k : 0;
    s_weight[threadIdx.x] =
        (sid >= 0 && mul_topk_weights) ? topk_weights[sid] : 1.0f;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_ksplit = groups_k16 / ksplit;
  // Split the K range over the warps *exactly*. A rank-local expert
  // intermediate is not always a multiple of SplitK*16 (TP4 pads 640/4=160 to
  // 192, i.e. 12 groups against 8 warps), and truncating the remainder would
  // silently drop K. Warps handed an empty range keep zeroed accumulators,
  // which the cross-warp reduction below absorbs.
  const int warp_begin = (warp * groups_per_ksplit) / SplitK;
  const int warp_end = ((warp + 1) * groups_per_ksplit) / SplitK;
  const int group_begin = blockIdx.z * groups_per_ksplit + warp_begin;
  const int group_end = blockIdx.z * groups_per_ksplit + warp_end;
  const size_t codes_per_expert = static_cast<size_t>(n) * (k >> 1);
  const size_t scales_per_expert = static_cast<size_t>(n) * groups_k16;
  const uint2* code_ptr =
      reinterpret_cast<const uint2*>(codes +
                                     static_cast<size_t>(expert) * codes_per_expert) +
      static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  const uint8_t* scale_ptr = group_scales +
                             static_cast<size_t>(expert) * scales_per_expert +
                             static_cast<size_t>(tile) * groups_k16 * 32 + lane;
  // w13 carries one global scale per expert for each of its two shards (gate,
  // up); a 32-column tile always lives inside one shard, so the shard follows
  // from the tile index. w2 passes gs_shards == 1.
  const float global_scale_value =
      global_scale[static_cast<size_t>(expert) * gs_shards +
                   (static_cast<int>(tile) < tiles_per_shard ? 0 : 1)];

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][chain][index] = 0.0f;
      }
    }
  }

#pragma unroll 4
  for (int group = group_begin; group < group_end; ++group) {
    const uint2 packed = __ldcs(code_ptr + static_cast<size_t>(group) * 32);
    const half2 scale = nvfp4_effective_scale(
        __ldg(scale_ptr + static_cast<size_t>(group) * 32), global_scale_value);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);

    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      const int slot = row_tile * kQpn2RowsPerCta + local_row;
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      if (s_expanded[slot] >= 0) {
        const half* input_row =
            input + static_cast<size_t>(s_token[slot]) *
                        static_cast<size_t>(ACT_INPUT ? act_stride : k);
        if constexpr (ACT_INPUT) {
          // w2 reads the w13 output directly: silu(gate) * up on the fly, so
          // the act_and_mul kernel and the cache2 round-trip disappear.
          input01 = qpn2_act_pair(
              *reinterpret_cast<const uint4*>(input_row + group * 16),
              *reinterpret_cast<const uint4*>(input_row + act_half +
                                              group * 16));
          input23 = qpn2_act_pair(
              *reinterpret_cast<const uint4*>(input_row + group * 16 + 8),
              *reinterpret_cast<const uint4*>(input_row + act_half +
                                              group * 16 + 8));
        } else {
          input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
          input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
        }
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input23);
      VLLM_SM70_QPN2_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      VLLM_SM70_QPN2_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][0][index] += accum[row_tile][chain][index];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][index];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    const int slot = element >> 5;
    const int expanded = s_expanded[slot];
    if (expanded < 0) {
      continue;
    }
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    value *= s_weight[slot];
    const size_t out_idx =
        static_cast<size_t>(expanded) * n + (static_cast<size_t>(tile) * 32 +
                                            (element & 31));
    if (ksplit == 1) {
      output[out_idx] = __float2half(value);
    } else {
      partials_out[static_cast<size_t>(blockIdx.z) * expanded_count * n +
                   out_idx] = value;
    }
  }
}

template <int SplitK, int NAcc, int RowTiles, bool ACT_INPUT = false>
void launch_qpn2_moe(const uint8_t* codes, const uint8_t* scales,
                     const half* input, half* output, float* partials_out,
                     const int32_t* sorted_token_ids,
                     const int32_t* expert_ids,
                     const int32_t* num_tokens_past_padded,
                     const float* topk_weights, const float* global_scale,
                     int n, int k, int m, int top_k, bool mul_topk_weights,
                     int route_tiles, int ksplit, int gs_shards,
                     int tiles_per_shard, int act_stride, int act_half,
                     cudaStream_t stream) {
  const dim3 grid(n / 32, route_tiles, ksplit);
  nvfp4_qpn2_moe_sm70_kernel<SplitK, NAcc, RowTiles, ACT_INPUT>
      <<<grid, (32 * SplitK), 0, stream>>>(
          codes, scales, input, output, partials_out, sorted_token_ids,
          expert_ids, num_tokens_past_padded, topk_weights, global_scale, n, k,
          m, top_k, mul_topk_weights, ksplit, gs_shards, tiles_per_shard,
          act_stride, act_half);
  if (ksplit > 1) {
    const int total = m * top_k * n;
    nvfp4_qpn2_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        partials_out, output, m * top_k, n, ksplit);
  }
}

// ---------------------------------------------------------------------------
// P6 shadow kernel: wide-column and/or k4-granularity route-tiled MoE GEMM.
//
// Attribution (2026-10-15 NCU,
// /data/sm70_results/round5/p6_moe/p6_moe_attribution.md): the route-tiled
// kernel above is L1-wavefront bound at prefill (72-83% L1/TEX, 33% of warp
// stalls = instruction queue full): every one of the n/32 column-tile CTAs
// re-gathers the same input rows, and at k=80 (gemm2) only 4 of the 8 K warps
// get a scale group, idling half the tensor cores.
//
// Two orthogonal levers, template-selected:
//   WCols  : CTA width in columns (32 = current; 160 = the row fragment is
//            loaded once and shared across the WCols/32 column sub-tiles ->
//            1/5 the global row wavefronts).
//   K4Gran : warp K-partition at 4-element (one mma m8n8k4) granularity
//            instead of 16-element scale-group granularity -> at k=80
//            (5 groups) every warp gets 2-3 slices instead of 4 warps
//            idling. A slice reuses the parent kernel's own MMA register
//            pair (MMA_j of the 4-per-group pattern uses A = {a[2j],
//            a[2j+1]}, B = {b[2j], b[2j+1]}), so no new dequant primitive.
//
// Shadow dispatch: default-off env gates keep the historical kernel
// bit-for-bit when unset. Each launcher logs its activation once to stderr
// (the G0 A/B harness keys off it to prove the compared path was live).
// G0 verdict (2026-10-15): the group-granularity wide path is bit-exact vs
// the historical kernel and 1.56x faster at the prefill w13 shape; the k4
// path is bit-exact when the k4 partition coincides with the group partition
// (k/16 % 8 == 0) but -9% at k=80, hence "auto" never selects k4.
// 0 = off, 2 = "auto" (k/16 >= 16, then first of {256, 128, 160, 64} that
// divides n), or the explicit width 64/128/160/256 (must divide n, else
// base).
int qpn2_moe_wide_cols() {
  static int v = -1;
  if (v < 0) {
    const char* env = std::getenv("VLLM_SM70_MOE_QPN2_WIDE");
    if (env != nullptr && std::strcmp(env, "auto") == 0) {
      v = 2;
    } else if (env != nullptr &&
               (std::strcmp(env, "64") == 0 || std::strcmp(env, "128") == 0 ||
                std::strcmp(env, "160") == 0 || std::strcmp(env, "256") == 0)) {
      v = std::atoi(env);
    } else {
      v = 0;
    }
  }
  return v;
}
int qpn2_moe_k4() {
  static int v = -1;
  if (v < 0) {
    const char* env = std::getenv("VLLM_SM70_MOE_QPN2_K4");
    if (env != nullptr && std::strcmp(env, "1") == 0) {
      v = 1;
    } else if (env != nullptr && std::strcmp(env, "auto") == 0) {
      v = 2;
    } else {
      v = 0;
    }
  }
  return v;
}
// T9 L-A research knob (2026-10-15, spec 2026-10-15-p6b-moe-t9-method.md):
// N-split warps on the 256-col wide path — each warp owns ONE 32-column
// sub-tile and runs the FULL K range (no SplitK, no partials smem, no
// cross-warp reduction epilogue). Replaces the k4 fix for the k<=128 shapes
// (k4 keeps all 8 warps busy but re-reads input fragments 4x, measured
// 0.54x). N-split: input loaded once per group per warp (same as wide),
// weight loads per CTA = kSubTiles (one warp each) instead of
// SplitK*kSubTiles (all warps, all sub-tiles). 1 = on (WCols=256 only).
int qpn2_moe_ns() {
  static int v = -1;
  if (v < 0) {
    const char* env = std::getenv("VLLM_SM70_MOE_QPN2_NS");
    v = (env != nullptr && std::strcmp(env, "1") == 0) ? 1 : 0;
  }
  return v;
}

template <int SplitK, int RowTiles, int WCols, bool K4Gran,
          bool NSplit = false, bool Fp16B = false>
__global__ void nvfp4_qpn2_moe_wide_kernel(
    const uint8_t* __restrict__ codes,   // [E, n, k/2] QPN2-packed
    const uint8_t* __restrict__ group_scales,  // [E, n, k/16]
    const half* __restrict__ input,      // [m, k]
    half* __restrict__ output,           // [m*top_k, n]
    float* __restrict__ partials_out,    // [ksplit, m*top_k, n] or null
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_past_padded,
    const float* __restrict__ topk_weights, const float* __restrict__ global_scale,
    int n, int k, int m, int top_k, bool mul_topk_weights, int ksplit,
    int gs_shards, int tiles_per_shard) {
  constexpr int kSubTiles = WCols / 32;
  constexpr int kMoeBlockM = RowTiles * kQpn2RowsPerCta;
  // NSplit: the partials smem is unused (no cross-warp reduction) — keep a
  // 1-row dummy so the array type stays non-empty.
  __shared__ float partials[NSplit ? 1 : SplitK][RowTiles * 256];
  __shared__ int s_expanded[kMoeBlockM];
  __shared__ int s_token[kMoeBlockM];
  __shared__ float s_weight[kMoeBlockM];

  const int moe_block = blockIdx.y;
  const int padded_tokens = num_tokens_past_padded[0];
  if (moe_block * kMoeBlockM >= padded_tokens) {
    return;
  }
  const int expert = expert_ids[moe_block];
  if (expert < 0) {
    return;
  }
  const int expanded_count = m * top_k;

  if (threadIdx.x < kMoeBlockM) {
    const int route_row = moe_block * kMoeBlockM + static_cast<int>(threadIdx.x);
    int sid = -1;
    if (route_row < padded_tokens) {
      const int cand = sorted_token_ids[route_row];
      if (cand >= 0 && cand < expanded_count) {
        sid = cand;
      }
    }
    s_expanded[threadIdx.x] = sid;
    s_token[threadIdx.x] = sid >= 0 ? sid / top_k : 0;
    s_weight[threadIdx.x] =
        (sid >= 0 && mul_topk_weights) ? topk_weights[sid] : 1.0f;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups_k16 = k >> 4;
  const int groups_per_ksplit = groups_k16 / ksplit;
  const size_t codes_per_expert = static_cast<size_t>(n) * (k >> 1);
  const size_t scales_per_expert = static_cast<size_t>(n) * groups_k16;
  const uint2* code_base =
      reinterpret_cast<const uint2*>(codes +
                                      static_cast<size_t>(expert) * codes_per_expert);
  const uint8_t* scale_base = group_scales +
                              static_cast<size_t>(expert) * scales_per_expert;
  // Fp16B: lane-major pre-dequantized B, [E, n/32, k/16, 32 lanes, 8 half2]
  // (the exact fragment layout dequant_e2m1x8 produced); `codes` aliases it.
  const half2* bf_base =
      Fp16B ? reinterpret_cast<const half2*>(
                   codes + static_cast<size_t>(expert) * (size_t)n * (k / 2))
            : nullptr;

  float accum[RowTiles][NSplit ? 1 : kSubTiles][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int t = 0; t < (NSplit ? 1 : kSubTiles); ++t) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[row_tile][t][index] = 0.0f;
      }
    }
  }

  if (!K4Gran || NSplit) {
    // Group granularity (the parent kernel's K partition), row fragments
    // loaded ONCE per group and shared across the column sub-tiles.
    // N-split: every warp covers the FULL K (no K split across warps);
    // the SplitK template dim only sizes the block (and the unused smem).
    const int warp_begin = NSplit ? 0 : (warp * groups_per_ksplit) / SplitK;
    const int warp_end =
        NSplit ? groups_per_ksplit : ((warp + 1) * groups_per_ksplit) / SplitK;
    const int group_begin = blockIdx.z * groups_per_ksplit + warp_begin;
    const int group_end = blockIdx.z * groups_per_ksplit + warp_end;
#pragma unroll 4
    for (int group = group_begin; group < group_end; ++group) {
      uint4 a[RowTiles][2];
#pragma unroll
      for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        const int slot = row_tile * kQpn2RowsPerCta + local_row;
        a[row_tile][0] = make_uint4(0, 0, 0, 0);
        a[row_tile][1] = make_uint4(0, 0, 0, 0);
        if (s_expanded[slot] >= 0) {
          const half* input_row = input + static_cast<size_t>(s_token[slot]) * k;
          a[row_tile][0] =
              *reinterpret_cast<const uint4*>(input_row + group * 16);
          a[row_tile][1] =
              *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
        }
      }
#pragma unroll
      for (int t = 0; t < (NSplit ? 1 : kSubTiles); ++t) {
        // N-split: this warp owns column sub-tile `warp`; the a-fragment
        // (already loaded per group) is shared across sub-tiles as before.
        const int st = NSplit ? warp : t;
        const int tidx = NSplit ? 0 : t;
        const int tile = blockIdx.x * kSubTiles + st;
        unsigned bwords[8];
        if constexpr (Fp16B) {
          // 2 x LDG.128 straight into the 8 b-fragment half2s: kills the
          // per-group prep (ldcs+ldg+scale+2 dequant ~22 instr -> 2).
          const half2* bb = bf_base +
              ((static_cast<size_t>(tile) * groups_k16 +
                static_cast<size_t>(group)) * 32 + lane) * 8;
          const uint4 q0 = *reinterpret_cast<const uint4*>(bb);
          const uint4 q1 = *reinterpret_cast<const uint4*>(bb + 4);
          bwords[0] = q0.x; bwords[1] = q0.y; bwords[2] = q0.z; bwords[3] = q0.w;
          bwords[4] = q1.x; bwords[5] = q1.y; bwords[6] = q1.z; bwords[7] = q1.w;
        } else {
          const uint2 packed =
              __ldcs(code_base + static_cast<size_t>(tile) * groups_k16 * 32 +
                     static_cast<size_t>(group) * 32 + lane);
          const float global_scale_value =
              global_scale[static_cast<size_t>(expert) * gs_shards +
                           (static_cast<int>(tile) < tiles_per_shard ? 0 : 1)];
          const half2 scale = nvfp4_effective_scale(
              __ldg(scale_base + static_cast<size_t>(tile) * groups_k16 * 32 +
                    static_cast<size_t>(group) * 32 + lane),
              global_scale_value);
          half2 weights[8];
          dequant_e2m1x8(packed.x, scale, weights);
          dequant_e2m1x8(packed.y, scale, weights + 4);
          const unsigned* w = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
          for (int i = 0; i < 8; ++i) bwords[i] = w[i];
        }
        const unsigned* b = bwords;
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
          const unsigned* a0 =
              reinterpret_cast<const unsigned*>(&a[row_tile][0]);
          const unsigned* a1 =
              reinterpret_cast<const unsigned*>(&a[row_tile][1]);
          VLLM_SM70_QPN2_MMA(accum[row_tile][tidx], a0[0], a0[1], b[0], b[1]);
          VLLM_SM70_QPN2_MMA(accum[row_tile][tidx], a0[2], a0[3], b[2], b[3]);
          VLLM_SM70_QPN2_MMA(accum[row_tile][tidx], a1[0], a1[1], b[4], b[5]);
          VLLM_SM70_QPN2_MMA(accum[row_tile][tidx], a1[2], a1[3], b[6], b[7]);
        }
      }
    }
  } else {
    // k4 granularity: the warp range is split over 4-element K slices. Each
    // slice is exactly one of the parent kernel's 4 per-group MMAs (same (a,
    // b) register pair), so all 8 warps stay busy at k=80 (5 scale groups).
    const int k4_per_ksplit = groups_per_ksplit * 4;
    const int k4_begin =
        blockIdx.z * k4_per_ksplit + (warp * k4_per_ksplit) / SplitK;
    const int k4_end =
        blockIdx.z * k4_per_ksplit + ((warp + 1) * k4_per_ksplit) / SplitK;
    half2 weights[kSubTiles][8];
    int cur_group = -1;
#pragma unroll 4
    for (int u4 = k4_begin; u4 < k4_end; ++u4) {
      const int group = u4 >> 2;
      const int j = u4 & 3;
      if (group != cur_group) {
        cur_group = group;
#pragma unroll
        for (int t = 0; t < kSubTiles; ++t) {
          const int tile = blockIdx.x * kSubTiles + t;
          const uint2 packed =
              __ldcs(code_base +
                     static_cast<size_t>(tile) * groups_k16 * 32 +
                     static_cast<size_t>(group) * 32 + lane);
          const float global_scale_value =
              global_scale[static_cast<size_t>(expert) * gs_shards +
                           (static_cast<int>(tile) < tiles_per_shard ? 0 : 1)];
          const half2 scale = nvfp4_effective_scale(
              __ldg(scale_base +
                    static_cast<size_t>(tile) * groups_k16 * 32 +
                    static_cast<size_t>(group) * 32 + lane),
              global_scale_value);
          dequant_e2m1x8(packed.x, scale, weights[t]);
          dequant_e2m1x8(packed.y, scale, weights[t] + 4);
        }
      }
#pragma unroll
      for (int t = 0; t < kSubTiles; ++t) {
        const unsigned* b = reinterpret_cast<const unsigned*>(&weights[t]);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
          const int slot = row_tile * kQpn2RowsPerCta + local_row;
          uint2 a = make_uint2(0, 0);
          if (s_expanded[slot] >= 0) {
            const half* input_row =
                input + static_cast<size_t>(s_token[slot]) * k;
            a = *reinterpret_cast<const uint2*>(input_row + group * 16 + 4 * j);
          }
          VLLM_SM70_QPN2_MMA(accum[row_tile][NSplit ? 0 : t], a.x, a.y, b[2 * j],
                             b[2 * j + 1]);
        }
      }
    }
  }

  // Epilogue: per column sub-tile, stage the warp partials, reduce across
  // warps, and write back (smem is reused per sub-tile). N-split skips all of
  // it: each warp owns one 32-column sub-tile and writes it directly (its own
  // accum, no cross-warp reduction, no smem sync).
#pragma unroll
  for (int t = 0; t < (NSplit ? 1 : kSubTiles); ++t) {
    const int tidx = NSplit ? 0 : t;
    const int st = NSplit ? warp : t;
    if (NSplit) {
      for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          // The C element (lane, index) belongs to tile row `output_row`
          // (mma C-fragment mapping, same as the partials epilogue) — NOT
          // the a-load row local_row; both the expanded id and the top-k
          // weight come from that row's slot.
          const int output_row =
              row_tile * kQpn2RowsPerCta + (index & 2) +
              ((lane & 16) ? 4 : 0) + (lane & 1);
          const int slot = output_row;
          const int expanded = s_expanded[slot];
          if (expanded < 0) {
            continue;
          }
          const int output_col =
              (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
          const float value =
              accum[row_tile][tidx][index] * s_weight[slot];
          const size_t out_idx =
              static_cast<size_t>(expanded) * n +
              (static_cast<size_t>(blockIdx.x) * kSubTiles + st) * 32 +
              quadpair * 8 + output_col;
          output[out_idx] = __float2half(value);
        }
      }
      continue;
    }
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        const int output_row = row_tile * kQpn2RowsPerCta + (index & 2) +
                               ((lane & 16) ? 4 : 0) + (lane & 1);
        const int output_col =
            (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
        partials[warp][output_row * 32 + quadpair * 8 + output_col] =
            accum[row_tile][tidx][index];
      }
    }
    __syncthreads();
    for (int element = threadIdx.x; element < RowTiles * 256;
         element += blockDim.x) {
      const int slot = element >> 5;
      const int expanded = s_expanded[slot];
      if (expanded < 0) {
        continue;
      }
      float value = 0.0f;
#pragma unroll
      for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
        value += partials[k_warp][element];
      }
      value *= s_weight[slot];
      const size_t out_idx =
          static_cast<size_t>(expanded) * n +
          (static_cast<size_t>(blockIdx.x) * WCols + t * 32 + (element & 31));
      if (ksplit == 1) {
        output[out_idx] = __float2half(value);
      } else {
        partials_out[static_cast<size_t>(blockIdx.z) * expanded_count * n +
                     out_idx] = value;
      }
    }
    __syncthreads();
  }
}

template <int SplitK, int RowTiles, int WCols, bool K4Gran,
          bool NSplit = false, bool Fp16B = false>
void launch_qpn2_moe_wide(const uint8_t* codes, const uint8_t* scales,
                          const half* input, half* output, float* partials_out,
                          const int32_t* sorted_token_ids,
                          const int32_t* expert_ids,
                          const int32_t* num_tokens_past_padded,
                          const float* topk_weights, const float* global_scale,
                          int n, int k, int m, int top_k, bool mul_topk_weights,
                          int route_tiles, int ksplit, int gs_shards,
                          int tiles_per_shard, cudaStream_t stream) {
  const int nx = n / WCols;
  // N-split contract: full-K per warp -> no outer K split, no reduce pass.
  if (NSplit) {
    ksplit = 1;
  }
  static bool logged = false;
  if (!logged) {
    logged = true;
    fprintf(stderr,
            "[sm70-qpn2] MoE shadow path active: cols=%d k4gran=%d nsplit=%d "
            "n=%d k=%d grid=(%d,%d,%d)\n",
            WCols, (int)K4Gran, (int)NSplit, n, k, nx, route_tiles, ksplit);
    fflush(stderr);
  }
  const dim3 grid(nx, route_tiles, ksplit);
  nvfp4_qpn2_moe_wide_kernel<SplitK, RowTiles, WCols, K4Gran, NSplit, Fp16B>
      <<<grid, (32 * SplitK), 0, stream>>>(
          codes, scales, input, output, partials_out, sorted_token_ids,
          expert_ids, num_tokens_past_padded, topk_weights, global_scale, n, k,
          m, top_k, mul_topk_weights, ksplit, gs_shards, tiles_per_shard);
  if (ksplit > 1) {
    const int total = m * top_k * n;
    nvfp4_qpn2_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        partials_out, output, m * top_k, n, ksplit);
  }
}

// One-time batched prepack of an [E, n, k/2] expert weight (plus its
// [E, n, k/16] E4M3 group scales) into the QPN2 lane-major layout, per expert.
__global__ void nvfp4_qpn2_prepack_moe_codes_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ weight, int e,
    int n, int k) {
  const size_t per_expert = static_cast<size_t>(n) * k / 2;
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(e) * per_expert;
  if (index >= numel) {
    return;
  }
  const size_t within = index % per_expert;
  const size_t expert = index / per_expert;
  const int physical_byte = static_cast<int>(within & 7);
  size_t outer = within >> 3;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  const int logical_k0 = group * 16 + qpn2_logical_k(physical_byte * 2);
  const int logical_k1 = group * 16 + qpn2_logical_k(physical_byte * 2 + 1);
  const int k_bytes = k >> 1;
  const uint8_t* src = weight + expert * static_cast<size_t>(n) * k_bytes;
  const uint8_t packed0 = src[static_cast<size_t>(column) * k_bytes + (logical_k0 >> 1)];
  const uint8_t packed1 = src[static_cast<size_t>(column) * k_bytes + (logical_k1 >> 1)];
  const uint8_t code0 =
      static_cast<uint8_t>((packed0 >> ((logical_k0 & 1) * 4)) & 0x0f);
  const uint8_t code1 =
      static_cast<uint8_t>((packed1 >> ((logical_k1 & 1) * 4)) & 0x0f);
  output[index] = static_cast<uint8_t>(code0 | (code1 << 4));
}

__global__ void nvfp4_qpn2_prepack_moe_scales_kernel(
    uint8_t* __restrict__ output, const uint8_t* __restrict__ scales, int e,
    int n, int k) {
  const size_t per_expert = static_cast<size_t>(n) * k / 16;
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t numel = static_cast<size_t>(e) * per_expert;
  if (index >= numel) {
    return;
  }
  const size_t within = index % per_expert;
  const size_t expert = index / per_expert;
  const int lane = static_cast<int>(within & 31);
  size_t outer = within >> 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int column = tile * 32 + qpn2_col_from_lane(lane);
  output[index] = scales[expert * static_cast<size_t>(n) * groups_k16 +
                         static_cast<size_t>(column) * groups_k16 + group];
}
}  // namespace qpn2

namespace {
inline void qpn2_check_launch() {
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 NVFP4 QPN2 launch failed: ",
                  cudaGetErrorString(err));
}
}  // namespace
// Persistent per-device float scratch for the K-split partials
// ([ksplit, m, n] row-major). Grown lazily to the largest shape seen so far;
// warmup covers the max before CUDA-graph capture, so the captured address is
// stable. If a grow is requested mid-capture, the caller falls back to
// ksplit=1 (checked before growth via cudaStreamIsCapturing).
struct Qpn2KsplitState {
  torch::stable::Tensor buf;
};

Qpn2KsplitState& qpn2_ksplit_state(int device) {
  static std::vector<Qpn2KsplitState*> states;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (static_cast<int>(device) >= static_cast<int>(states.size())) {
    states.resize(device + 1, nullptr);
  }
  if (!states[device]) {
    states[device] = new Qpn2KsplitState();
  }
  return *states[device];
}

// A/B knob: VLLM_SM70_QPN2_KSPLIT=<int> caps ksplit (0 = K-split off).
// Unset = 0 (off): measured flat vs ksplit=1 on all model shapes, so the
// legacy path stays the default. Set e.g. VLLM_SM70_QPN2_KSPLIT=8 to enable.
int qpn2_ksplit_cap() {
  static int cap = -1;
  if (cap < 0) {
    const char* env = std::getenv("VLLM_SM70_QPN2_KSPLIT");
    cap = (env != nullptr) ? std::atoi(env) : 0;
    if (cap < 0) cap = 0;
  }
  return cap;
}


std::vector<torch::stable::Tensor> sm70_nvfp4_qpn2_prepare(
    torch::stable::Tensor& weight, torch::stable::Tensor& weight_scale) {
  torch::stable::accelerator::DeviceGuard device_guard(
      weight.get_device_index());
  int64_t n = weight.size(0);
  int64_t k = weight.size(1) * 2;
  STD_TORCH_CHECK(n % 32 == 0 && k % 16 == 0,
                  "SM70 NVFP4 QPN2 prepare needs n%32==0 and k%16==0, got n=",
                  n, " k=", k);
  int threads = static_cast<int>(qpn2::kPrepareThreads);
  cudaStream_t stream = get_current_cuda_stream(weight.get_device_index());
  const uint8_t* w = reinterpret_cast<const uint8_t*>(weight.data_ptr());
  const uint8_t* s = reinterpret_cast<const uint8_t*>(weight_scale.data_ptr());

  std::vector<int64_t> codes_shape = {n, k / 2};
  torch::headeronly::IntHeaderOnlyArrayRef codes_sizes(codes_shape.data(),
                                                       codes_shape.size());
  auto codes = torch::stable::new_empty(
      weight, codes_sizes, torch::headeronly::ScalarType::Byte);
  int code_blocks = static_cast<int>((n * k / 2 + threads - 1) / threads);
  qpn2::nvfp4_qpn2_prepack_codes_kernel<<<code_blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t*>(codes.mutable_data_ptr()), w,
      static_cast<int>(n), static_cast<int>(k));

  std::vector<int64_t> scales_shape = {n, k / 16};
  torch::headeronly::IntHeaderOnlyArrayRef scales_sizes(scales_shape.data(),
                                                        scales_shape.size());
  auto scales = torch::stable::new_empty(
      weight, scales_sizes, torch::headeronly::ScalarType::Byte);
  int scale_blocks = static_cast<int>((n * k / 16 + threads - 1) / threads);
  qpn2::nvfp4_qpn2_prepack_scales_kernel<<<scale_blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t*>(scales.mutable_data_ptr()), s,
      static_cast<int>(n), static_cast<int>(k));
  qpn2_check_launch();
  return {codes, scales};
}

void sm70_nvfp4_qpn2_gemm(torch::stable::Tensor& out,
                          torch::stable::Tensor& input,
                          torch::stable::Tensor& codes,
                          torch::stable::Tensor& scales, double global_scale,
                          int64_t split_k, int64_t accumulator_chains) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  STD_TORCH_CHECK(split_k == 8 || split_k == 16 || split_k == 32,
                  "SM70 NVFP4 QPN2 split_k must be 8, 16, or 32");
  STD_TORCH_CHECK(accumulator_chains == 1 || accumulator_chains == 2,
                  "SM70 NVFP4 QPN2 accumulator_chains must be 1 or 2");
  int n = static_cast<int>(out.size(1));
  int k = static_cast<int>(input.size(1));
  int m = static_cast<int>(input.size(0));
  STD_TORCH_CHECK(k / 16 % split_k == 0,
                  "SM70 NVFP4 QPN2 requires k/16 divisible by split_k");
  const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  const uint8_t* scale_ptr =
      reinterpret_cast<const uint8_t*>(scales.data_ptr());
  const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr());
  half* output_ptr = reinterpret_cast<half*>(out.mutable_data_ptr());
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  float gs = static_cast<float>(global_scale);

  // SM-saturating K-split: small-N GEMMs launch too few CTAs to keep the 80
  // SMs busy (e.g. n=2560 -> 80 CTAs -> ~10% occupancy -> ~1/3 DRAM peak).
  // Split K across grid-z so CTA count ~= 5120/split_k (8 CTAs of 256 threads
  // per SM); the per-slice float partials are reduced by a follow-up kernel.
  // ksplit must divide k/16/split_k.
  const int gk = k / 16;
  const int maxks = gk / static_cast<int>(split_k);
  const int nctas = n / 32;
  int ksplit = 1;
  {
    // target CTA count = 5120/split_k (8 CTAs/SM of (32*split_k) threads)
    const int target = 5120 / static_cast<int>(split_k);
    int want = (target + nctas - 1) / nctas;
    if (want > 8) want = 8;
    if (want > maxks) want = maxks;
    for (int d = want; d > 1; --d) {
      if (gk % (static_cast<int>(split_k) * d) == 0) {
        ksplit = d;
        break;
      }
    }
    const int cap = qpn2_ksplit_cap();
    if (cap == 0) {
      ksplit = 1;
    } else if (ksplit > cap) {
      // clamp to the largest divisor <= cap
      ksplit = 1;
      for (int d = std::min(cap, maxks); d > 1; --d) {
        if (gk % (static_cast<int>(split_k) * d) == 0) {
          ksplit = d;
          break;
        }
      }
    }
  }
  float* partials_ptr = nullptr;
  if (ksplit > 1) {
    auto& st = qpn2_ksplit_state(input.get_device_index());
    const int64_t need = static_cast<int64_t>(ksplit) * m * n;
    if (st.buf.defined() && st.buf.numel() >= need) {
      partials_ptr = reinterpret_cast<float*>(st.buf.mutable_data_ptr());
    } else {
      // Grow only when not mid-capture; otherwise fall back to no K-split
      // (a reallocation would invalidate an in-flight captured graph).
      cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &cap);
      if (cap == cudaStreamCaptureStatusNone) {
        std::vector<int64_t> shape = {need};
        torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(),
                                                       shape.size());
        st.buf = torch::stable::new_empty(input, sizes,
                                          torch::headeronly::ScalarType::Float);
      } else {
        ksplit = 1;
      }
      partials_ptr =
          (ksplit > 1) ? reinterpret_cast<float*>(st.buf.mutable_data_ptr())
                       : nullptr;
    }
  }

#define LAUNCH_QPN2(ROWS, SPLIT, NACC) \
      qpn2::launch_qpn2<SPLIT, NACC, ROWS>(code_ptr, scale_ptr, input_ptr, \
                                          output_ptr, partials_ptr, n, k, m, \
                                          gs, ksplit, stream)
  bool two_tile = (m > qpn2::kQpn2RowsPerCta) && (m <= 16);
  if (two_tile && split_k == 8 && accumulator_chains == 1) {
    LAUNCH_QPN2(2, 8, 1);
  } else if (two_tile && split_k == 8) {
    LAUNCH_QPN2(2, 8, 2);
  } else if (two_tile && split_k == 16 && accumulator_chains == 1) {
    LAUNCH_QPN2(2, 16, 1);
  } else if (two_tile && split_k == 16) {
    LAUNCH_QPN2(2, 16, 2);
  } else if (split_k == 8 && accumulator_chains == 1) {
    LAUNCH_QPN2(1, 8, 1);
  } else if (split_k == 8) {
    LAUNCH_QPN2(1, 8, 2);
  } else if (split_k == 16 && accumulator_chains == 1) {
    LAUNCH_QPN2(1, 16, 1);
  } else if (split_k == 16) {
    LAUNCH_QPN2(1, 16, 2);
  } else if (accumulator_chains == 1) {
    LAUNCH_QPN2(1, 32, 1);
  } else {
    LAUNCH_QPN2(1, 32, 2);
  }
#undef LAUNCH_QPN2
  qpn2_check_launch();
}

std::vector<torch::stable::Tensor> sm70_nvfp4_qpn2_prepare_moe(
    torch::stable::Tensor& weight, torch::stable::Tensor& weight_scale) {
  torch::stable::accelerator::DeviceGuard device_guard(
      weight.get_device_index());
  STD_TORCH_CHECK(weight.dim() == 3 && weight_scale.dim() == 3,
                  "SM70 NVFP4 QPN2 MoE prepare needs [E, n, k/2] and "
                  "[E, n, k/16] tensors");
  const int64_t e = weight.size(0);
  const int64_t n = weight.size(1);
  const int64_t k = weight.size(2) * 2;
  STD_TORCH_CHECK(n % 32 == 0 && k % 16 == 0,
                  "SM70 NVFP4 QPN2 MoE prepare needs n%32==0 and k%16==0, got "
                  "n=",
                  n, " k=", k);
  STD_TORCH_CHECK(weight_scale.size(0) == e && weight_scale.size(1) == n &&
                      weight_scale.size(2) == k / 16,
                  "SM70 NVFP4 QPN2 MoE prepare scale shape mismatch");
  const int threads = static_cast<int>(qpn2::kPrepareThreads);
  cudaStream_t stream = get_current_cuda_stream(weight.get_device_index());
  const uint8_t* w = reinterpret_cast<const uint8_t*>(weight.data_ptr());
  const uint8_t* s = reinterpret_cast<const uint8_t*>(weight_scale.data_ptr());

  std::vector<int64_t> codes_shape = {e, n, k / 2};
  torch::headeronly::IntHeaderOnlyArrayRef codes_sizes(codes_shape.data(),
                                                       codes_shape.size());
  auto codes = torch::stable::new_empty(
      weight, codes_sizes, torch::headeronly::ScalarType::Byte);
  const int64_t code_numel = e * n * k / 2;
  qpn2::nvfp4_qpn2_prepack_moe_codes_kernel<<<
      static_cast<int>((code_numel + threads - 1) / threads), threads, 0,
      stream>>>(reinterpret_cast<uint8_t*>(codes.mutable_data_ptr()), w,
                static_cast<int>(e), static_cast<int>(n), static_cast<int>(k));

  std::vector<int64_t> scales_shape = {e, n, k / 16};
  torch::headeronly::IntHeaderOnlyArrayRef scales_sizes(scales_shape.data(),
                                                        scales_shape.size());
  auto scales = torch::stable::new_empty(
      weight, scales_sizes, torch::headeronly::ScalarType::Byte);
  const int64_t scale_numel = e * n * k / 16;
  qpn2::nvfp4_qpn2_prepack_moe_scales_kernel<<<
      static_cast<int>((scale_numel + threads - 1) / threads), threads, 0,
      stream>>>(reinterpret_cast<uint8_t*>(scales.mutable_data_ptr()), s,
                static_cast<int>(e), static_cast<int>(n), static_cast<int>(k));
  qpn2_check_launch();
  return {codes, scales};
}

// A/B knob: warps per CTA of the route-tiled MoE kernel (unset = 8, the
// historical value; 16 doubles the in-flight loads per lane-group, which is the
// axis the dense QPN2 kernel reaches 690 GB/s on and this one only 293).
int qpn2_moe_warps() {
  static int w = -1;
  if (w < 0) {
    const char* env = std::getenv("VLLM_SM70_MOE_QPN2_WARPS");
    w = (env != nullptr) ? std::atoi(env) : 8;
    if (w != 16) w = 8;
  }
  return w;
}

// A/B knob for the route-tiled MoE kernel's cross-CTA K-split cap.
// Unset = 1 (off): once the empty route tiles are gone from the launch, one
// K-split-free CTA per (column tile, route tile) measured better than splitting
// - at 1 token / 8 experts the expert gemms run 13.7 + 12.9 us per layer against
// 16.3 + 13.8 with the CTA-count heuristic in charge (cap 8) and 15.9 + 12.5 at
// cap 16. Set VLLM_SM70_MOE_QPN2_KSPLIT=8 to hand the decision back to it.
int qpn2_moe_ksplit_cap() {
  static int cap = -1;
  if (cap < 0) {
    const char* env = std::getenv("VLLM_SM70_MOE_QPN2_KSPLIT");
    cap = (env != nullptr) ? std::atoi(env) : 1;
    if (cap < 1) cap = 1;
  }
  return cap;
}

template <bool Fp16B>
void sm70_nvfp4_qpn2_moe_gemm_impl(torch::stable::Tensor& out,
                              torch::stable::Tensor& input,
                              torch::stable::Tensor& codes,
                              torch::stable::Tensor& scales,
                              torch::stable::Tensor& global_scale,
                              torch::stable::Tensor& sorted_token_ids,
                              torch::stable::Tensor& expert_ids,
                              torch::stable::Tensor& num_tokens_past_padded,
                              torch::stable::Tensor& topk_weights,
                              int64_t top_k, bool mul_topk_weights,
                              int64_t num_shards, int64_t moe_block_size,
                              bool fuse_act) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  const int m = static_cast<int>(input.size(0));
  const int experts = static_cast<int>(codes.size(0));
  STD_TORCH_CHECK(out.size(0) == static_cast<int64_t>(m) * top_k,
                  "SM70 NVFP4 QPN2 MoE expects out rows == m * top_k, got ",
                  out.size(0), " for m=", m, " top_k=", top_k);
  if constexpr (!Fp16B) {
    STD_TORCH_CHECK(codes.size(1) == n && codes.size(2) == k / 2,
                    "SM70 NVFP4 QPN2 MoE code layout mismatch: need [E,", n,
                    ",", k / 2, "]");
    STD_TORCH_CHECK(scales.size(1) == n && scales.size(2) == k / 16,
                    "SM70 NVFP4 QPN2 MoE scale layout mismatch");
  } else {
    STD_TORCH_CHECK(codes.size(1) == n / 32 && codes.size(2) == k / 16 &&
                       codes.size(3) == 32 && codes.size(4) == 16,
                    "SM70 NVFP4 QPN2 MoE fp16 B layout mismatch: need [E, ",
                    n / 32, ",", k / 16, ", 32, 16]");
  }
  STD_TORCH_CHECK(num_shards == 1 || num_shards == 2,
                  "SM70 NVFP4 QPN2 MoE num_shards must be 1 (w2) or 2 (w13)");
  STD_TORCH_CHECK(n % num_shards == 0,
                  "SM70 NVFP4 QPN2 MoE needs n divisible by num_shards");
  // An NVFP4 expert tensor carries ONE global scale per expert (modelopt
  // stores w13_weight_scale_2 as [E], gate and up included), so the shard
  // axis only appears if the caller really supplies E * num_shards values.
  const int64_t gs_numel = global_scale.numel();
  const int gs_shards = gs_numel == static_cast<int64_t>(experts) * num_shards
                            ? static_cast<int>(num_shards)
                            : 1;
  STD_TORCH_CHECK(gs_numel == static_cast<int64_t>(experts) * gs_shards,
                  "SM70 NVFP4 QPN2 MoE needs one global scale per expert"
                  " (or per expert and w13 shard): expected ",
                  experts, " or ", experts * num_shards, ", got ", gs_numel);
  STD_TORCH_CHECK(n % 32 == 0 && k % 16 == 0,
                  "SM70 NVFP4 QPN2 MoE needs n%32==0 and k%16==0");
  STD_TORCH_CHECK(input.stride(1) == 1,
                  "SM70 NVFP4 QPN2 MoE needs a row-contiguous input");
  // Fused activation (w2 only): `input` is a [rows, k] view of the w13 output
  // whose row stride is 2*k; gate lives at +0, up at +k.
  const int act_stride = fuse_act ? static_cast<int>(input.stride(0)) : 0;
  const int act_half = fuse_act ? k : 0;
  if (fuse_act) {
    STD_TORCH_CHECK(num_shards == 1,
                    "SM70 NVFP4 QPN2 MoE fused activation is w2-only");
    STD_TORCH_CHECK(input.stride(0) == 2 * static_cast<int64_t>(k),
                    "SM70 NVFP4 QPN2 MoE fused activation needs a 2k row "
                    "stride (the w13 gate/up buffer)");
  }
  STD_TORCH_CHECK(out.stride(1) == 1,
                  "SM70 NVFP4 QPN2 MoE needs a row-contiguous output");
  // sorted_token_ids is a CAPACITY buffer (num_tokens + E * (block - 1)), not
  // a padded-to-block length; the used part is num_tokens_post_padded[0], which
  // the kernel reads on device to retire the extra tiles (same as the Marlin MoE
  // route-tile count, which also derives from numel()).
  STD_TORCH_CHECK(moe_block_size == 8 || moe_block_size == 16,
                  "SM70 NVFP4 QPN2 MoE supports route blocks of 8 or 16 slots");
  const int64_t sorted_numel = sorted_token_ids.numel();
  const int capacity_tiles =
      static_cast<int>((sorted_numel + moe_block_size - 1) / moe_block_size);
  // A route block only exists for an expert that received slots, and each such
  // expert costs ceil(count / block) blocks. With slots = m * top_k that is at
  // min(E, slots) + slots / block blocks however the slots are distributed, so
  // tiles past that bound are all-padding and cannot hold output. Launching the
  // capacity instead is what makes the skinny gemm2 slow: at 1 token / 8 of 256
  // experts / block 8 the capacity is 225 tiles per column but only 9 can be
  // live, and 96% of the CTAs of every launch do nothing but read the padded
  // count and retire.
  const int64_t slots = static_cast<int64_t>(m) * top_k;
  const int64_t live_tiles =
      std::min<int64_t>(experts, slots) + slots / moe_block_size + 1;
  const int route_tiles =
      static_cast<int>(std::min<int64_t>(capacity_tiles, live_tiles));
  STD_TORCH_CHECK(expert_ids.numel() >= capacity_tiles,
                  "SM70 NVFP4 QPN2 MoE expert_ids shorter than the route tiles");

  constexpr int kSplitK = 8;
  const int groups_k16 = k / 16;
  // The kernel splits K over its warps with an exact remainder distribution, so
  // a rank-local intermediate that is not a multiple of kSplitK*16 (TP4: 192)
  // is supported; only whole 16-value scale groups are required, which the
  // prepack already guarantees.
  STD_TORCH_CHECK(groups_k16 > 0 && k % 16 == 0,
                  "SM70 NVFP4 QPN2 MoE needs k % 16 == 0, got k=", k);
  const int nctas = (n / 32) * route_tiles;
  const int target = 5120 / kSplitK;
  // K-split partials are fp32 [ksplit, m*top_k, n]. Split-K exists to buy CTA
  // parallelism for decode-sized problems: at prefill sizes the route tiles
  // already fill the SMs, and the same buffer would cost hundreds of MB of
  // resident scratch, so cap the split by bytes as well as by K divisors.
  const int64_t per_split_bytes =
      static_cast<int64_t>(m) * top_k * n * static_cast<int64_t>(sizeof(float));
  const int64_t ks_by_mem =
      (32LL << 20) / std::max<int64_t>(per_split_bytes, 1);
  int ksplit = 1;
  {
    int want = (target + nctas - 1) / nctas;
    // Outer split-K only needs to divide the group count; the warps inside each
    // z-slice split their share exactly.
    int maxks = groups_k16;
    if (maxks > ks_by_mem) maxks = static_cast<int>(ks_by_mem);
    if (want > 8) want = 8;
    if (want > maxks) want = maxks;
    const int cap = qpn2_moe_ksplit_cap();
    if (want > cap) want = cap;
    for (int d = want; d > 1; --d) {
      if (groups_k16 % d == 0) {
        ksplit = d;
        break;
      }
    }
  }

  float* partials_ptr = nullptr;
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  if (ksplit > 1) {
    auto& st = qpn2_ksplit_state(input.get_device_index());
    const int64_t need =
        static_cast<int64_t>(ksplit) * m * top_k * n;
    if (st.buf.defined() && st.buf.numel() >= need) {
      partials_ptr = reinterpret_cast<float*>(st.buf.mutable_data_ptr());
    } else {
      cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &cap);
      if (cap == cudaStreamCaptureStatusNone) {
        std::vector<int64_t> shape = {need};
        torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(),
                                                       shape.size());
        st.buf = torch::stable::new_empty(input, sizes,
                                          torch::headeronly::ScalarType::Float);
        partials_ptr = reinterpret_cast<float*>(st.buf.mutable_data_ptr());
      } else {
        ksplit = 1;
      }
    }
  }

  // A 16-slot route block makes the mma compute 16 rows per expert; at decode
  // batch sizes (<= 8 tokens x top_k routes spread over up to 256 experts) a
  // block holds one to a few real slots, and that padding - not DRAM - is what
  // bounds this kernel: 16.9 us/launch at 16 slots against the 6.1 us its
  // 4.2 MB of weights would take at this box's 691 GB/s, and split-K beyond the
  // SM-filling point changes nothing. 8 slots halves the wasted rows.
#define LAUNCH_MOE_ACT(ROWS, ACTF)                                         \
  qpn2::launch_qpn2_moe<kSplitK, 1, ROWS, ACTF>(                            \
      reinterpret_cast<const uint8_t*>(codes.data_ptr()),                    \
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),                   \
      reinterpret_cast<const half*>(input.data_ptr()),                       \
      reinterpret_cast<half*>(out.mutable_data_ptr()), partials_ptr,         \
      sorted_token_ids.const_data_ptr<int32_t>(),                           \
      expert_ids.const_data_ptr<int32_t>(),                                 \
      num_tokens_past_padded.const_data_ptr<int32_t>(),                     \
      topk_weights.const_data_ptr<float>(),                                 \
      global_scale.const_data_ptr<float>(), n, k, m, static_cast<int>(top_k),\
      mul_topk_weights, route_tiles, ksplit, gs_shards,                      \
      (n / gs_shards) / 32, act_stride, act_half, stream)
  // 16 warps per CTA only pays when every warp still gets a whole number of
  // 16-value K groups; otherwise fall back to the 8-warp instantiation.
  const bool wide = qpn2_moe_warps() == 16 && ((groups_k16 / ksplit) % 16 == 0) &&
                    act_stride == 0;
#define LAUNCH_MOE_W_ACT(W, ROWS, ACTF)                                      \
  qpn2::launch_qpn2_moe<W, 1, ROWS, ACTF>(                                    \
      reinterpret_cast<const uint8_t*>(codes.data_ptr()),                      \
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),                     \
      reinterpret_cast<const half*>(input.data_ptr()),                         \
      reinterpret_cast<half*>(out.mutable_data_ptr()), partials_ptr,           \
      sorted_token_ids.const_data_ptr<int32_t>(),                             \
      expert_ids.const_data_ptr<int32_t>(),                                   \
      num_tokens_past_padded.const_data_ptr<int32_t>(),                       \
      topk_weights.const_data_ptr<float>(),                                   \
      global_scale.const_data_ptr<float>(), n, k, m, static_cast<int>(top_k),  \
      mul_topk_weights, route_tiles, ksplit, gs_shards,                        \
      (n / gs_shards) / 32, act_stride, act_half, stream)
#define LAUNCH_MOE(ROWS)                                                     \
  do {                                                                       \
    if (act_stride > 0) {                                                    \
      LAUNCH_MOE_ACT(ROWS, true);                                            \
    } else {                                                                 \
      LAUNCH_MOE_ACT(ROWS, false);                                           \
    }                                                                        \
  } while (0)
#define LAUNCH_MOE_W(W, ROWS)                                                \
  do {                                                                       \
    if (act_stride > 0) {                                                    \
      LAUNCH_MOE_W_ACT(W, ROWS, true);                                       \
    } else {                                                                 \
      LAUNCH_MOE_W_ACT(W, ROWS, false);                                      \
    }                                                                        \
  } while (0)
  const bool block8 = (moe_block_size == 8);
  #define LAUNCH_WIDE_T(ROWS, WC, K4G)                                       \
    qpn2::launch_qpn2_moe_wide<kSplitK, ROWS, WC, K4G, false, Fp16B>(         \
        reinterpret_cast<const uint8_t*>(codes.data_ptr()),                   \
        reinterpret_cast<const uint8_t*>(scales.data_ptr()),                  \
        reinterpret_cast<const half*>(input.data_ptr()),                      \
        reinterpret_cast<half*>(out.mutable_data_ptr()), partials_ptr,         \
        sorted_token_ids.const_data_ptr<int32_t>(),                           \
        expert_ids.const_data_ptr<int32_t>(),                                 \
        num_tokens_past_padded.const_data_ptr<int32_t>(),                     \
        topk_weights.const_data_ptr<float>(),                                \
        global_scale.const_data_ptr<float>(), n, k, m, static_cast<int>(top_k),\
        mul_topk_weights, route_tiles, ksplit, gs_shards,                     \
        (n / gs_shards) / 32, stream)
  // T9 L-A (2026-10-15): N-split warps on the 256-col path — each warp owns one
  // 32-column sub-tile, full K, ksplit=1, no partials/reduction. Research
  // knob VLLM_SM70_MOE_QPN2_NS (default off).
  #define LAUNCH_WIDE_NS(ROWS)                                               \
    qpn2::launch_qpn2_moe_wide<kSplitK, ROWS, 256, false, true, Fp16B>(        \
        reinterpret_cast<const uint8_t*>(codes.data_ptr()),                   \
        reinterpret_cast<const uint8_t*>(scales.data_ptr()),                  \
        reinterpret_cast<const half*>(input.data_ptr()),                      \
        reinterpret_cast<half*>(out.mutable_data_ptr()), partials_ptr,         \
        sorted_token_ids.const_data_ptr<int32_t>(),                           \
        expert_ids.const_data_ptr<int32_t>(),                                 \
        num_tokens_past_padded.const_data_ptr<int32_t>(),                     \
        topk_weights.const_data_ptr<float>(),                                \
        global_scale.const_data_ptr<float>(), n, k, m, static_cast<int>(top_k),\
        mul_topk_weights, route_tiles, ksplit, gs_shards,                     \
        (n / gs_shards) / 32, stream)
  // P6 shadow path (T8, 2026-10-15): default off; env gates and kernel docs on
  // nvfp4_qpn2_moe_wide_kernel. wcols>0 wins over k4mode (both = widek4).
  int wcols = 0;
  int k4mode = 0;
  const int env_wide_diag = qpn2::qpn2_moe_wide_cols();
  const int env_k4_diag = qpn2::qpn2_moe_k4();
  const int env_ns = qpn2::qpn2_moe_ns();
  // T9 final-state integration (#193, 2026-10-15): auto mode also routes the
  // small-k (groups_k16 <= 8, i.e. k<=128) shapes to the 256-col NS path —
  // the T9 in-service drop-in (WIDE=256 + NS=1) measured 1.82x on w2, while
  // the K-split warps idle there. Large-k shapes (groups >= 16) stay on
  // K-split (NS loses to it, T9 L-B).
  const bool ns_auto = env_ns == 1 || (env_wide_diag == 2 && groups_k16 <= 8);
  {
    const int env_wide = env_wide_diag;
    // auto (2): widest CTA width that divides n. 128 first: it covers the
    // production shapes (w13 n=256 — moe_intermediate_size 640 padded to 1024
    // then split TP8 -> 2x128; w2 n=2560), 160 keeps the offline-bench
    // shape alive, 64 is the narrow fallback.
    if (env_wide == 2) {
      // auto never widens the k<256 (gemm2-like) shapes: they are
      // epilogue/latency bound, and measuring wide there is -24% (G0
      // 2026-10-15, k=128). Only the k>=256 (gemm1-like) shapes win.
      if (groups_k16 >= 16) {
        static const int kWideCand[] = {256, 128, 160, 64};
        for (const int c : kWideCand) {
          if (n % c == 0) {
            wcols = c;
            break;
          }
        }
      } else if (groups_k16 <= 8 && ns_auto && n % 256 == 0) {
        // T9 final-state (#193): small-k (k<=128) shapes take the 256-col
        // NS path (the dispatch below routes NS for them): K-split warps
        // idle there and NS measured 1.82x in-service on w2 (ruler
        // 4158->4466). The CTA-starvation guard below still vetoes
        // decode-scale launches.
        wcols = 256;
      }
    } else if (env_wide != 0 && n % env_wide == 0) {
      wcols = env_wide;
    }
    const int env_k4 = env_k4_diag;
    // "auto" deliberately never enables k4: at the k=80 gemm2 shape the k4
    // partition measures -9% (the kernel is epilogue/latency-bound, not
    // TC-bound there — G0 2026-10-15). "1" remains a manual research knob.
    if (env_k4 == 1) {
      k4mode = 1;
    }
  }
  // The wide CTA must not starve the SMs: at WCols=256 the grid.x collapses
  // to n/256 and at decode scale (small route_tiles) the total CTA count
  // under-fills the 80 SMs -> measured -12% decode e2e (G1 2026-10-15).
  // Fall back to the historical kernel whenever the wide launch would
  // produce fewer than ~160 CTAs (2/SM). Prefill (route_tiles ~5k) always
  // passes; decode (route_tiles ~70) does not at n=256.
  if (wcols > 0 &&
      static_cast<int64_t>(n / wcols) * route_tiles < 160) {
    wcols = 0;
  }
  // The k=1 speculative verify captures its MoE GEMMs with route_tiles == 20
  // (two rows x ten experts); the wide CTA measured slower there (MTP step
  // 23.3 -> 21.0 ms with the wide path off), while prefill (route_tiles >= 1025)
  // and the other decode capture sizes (>= 30) keep it. The dispatch only runs
  // at graph-capture time, so the choice is baked into the verify's graph.
  if (wcols > 0 && route_tiles <= 20) {
    wcols = 0;
  }
  if (Fp16B) {
    STD_TORCH_CHECK(wcols > 0,
                    "SM70 NVFP4 QPN2 MoE fp16 path requires the wide kernel "
                    "(route_tiles too small for this call)");
  }
  if (std::getenv("VLLM_SM70_MOE_DIAG_ALL") != nullptr) {
    FILE* df = std::fopen("/tmp/sm70_qpn2_all", "a");
    if (df != nullptr) {
      std::fprintf(df,
                   "m=%d n=%d k=%d route_tiles=%d groups=%d wcols=%d path=%s\n",
                   m, n, k, route_tiles, groups_k16, wcols,
                   wcols > 0 ? "wide" : (k4mode ? "k4" : (wide ? "wide16" : "base")));
      std::fclose(df);
    }
  }
  // One-shot dispatch diagnostic (G1 2026-10-15): the engine swallows worker
  // stderr, so the decision must land in a file to prove which path a
  // production process actually took.
  static bool diag_done = false;
  if (!diag_done) {
    diag_done = true;
    FILE* f = std::fopen("/tmp/sm70_qpn2_moe_diag", "a");
    if (f != nullptr) {
      const char* raw_env = std::getenv("VLLM_SM70_MOE_QPN2_WIDE");
      fprintf(f,
              "pid=%d raw_wide_env=%s env_wide=%d env_k4=%d env_ns=%d "
              "groups=%d n=%d wcols=%d k4mode=%d block8=%d path=%s\n",
              getpid(), raw_env != nullptr ? raw_env : "NULL", env_wide_diag,
              env_k4_diag, env_ns, groups_k16, n, wcols, k4mode, (int)block8,
              wcols > 0 ? "wide" : (k4mode ? "k4" : (wide ? "wide16" : "base")));
      std::fclose(f);
    }
  }
  if (wcols == 160) {
    if (block8) {
      if (k4mode) {
        LAUNCH_WIDE_T(1, 160, true);
      } else {
        LAUNCH_WIDE_T(1, 160, false);
      }
    } else {
      if (k4mode) {
        LAUNCH_WIDE_T(2, 160, true);
      } else {
        LAUNCH_WIDE_T(2, 160, false);
      }
    }
  } else if (wcols == 256) {
    // T9 L-A gate data (ab_ns, 2026-10-15): N-split wins at k<=128
    // (groups_k16<=8 — the shapes where the K-split warps idle; w2
    // measures 1.81x at m=8192) but LOSES to K-split at k=2560
    // (w13: NS 3.859 vs wide256 2.982 ms = 0.77x at m=8192): every NS
    // warp re-reads its weight sub-tile over the FULL K, so large-k
    // shapes need the K split. Never route them to NS.
    const bool ns_eligible = ns_auto && groups_k16 <= 8;
    if (ns_eligible) {
      // T9 L-A (research): N-split on the 256-col path.
      if (block8) {
        LAUNCH_WIDE_NS(1);
      } else {
        LAUNCH_WIDE_NS(2);
      }
    } else {
      // group granularity only (k4 instantiations stay at 32/160)
      if (block8) {
        LAUNCH_WIDE_T(1, 256, false);
      } else {
        LAUNCH_WIDE_T(2, 256, false);
      }
    }
  } else if (wcols == 128) {
    // group granularity only at 128 (k4 instantiations stay at 32/160)
    if (block8) {
      LAUNCH_WIDE_T(1, 128, false);
    } else {
      LAUNCH_WIDE_T(2, 128, false);
    }
  } else if (wcols == 64) {
    if (block8) {
      LAUNCH_WIDE_T(1, 64, false);
    } else {
      LAUNCH_WIDE_T(2, 64, false);
    }
  } else if (k4mode) {
    if (block8) {
      LAUNCH_WIDE_T(1, 32, true);
    } else {
      LAUNCH_WIDE_T(2, 32, true);
    }
  } else if (wide) {
    if (block8) {
      LAUNCH_MOE_W(16, 1);
    } else {
      LAUNCH_MOE_W(16, 2);
    }
  } else if (block8) {
    LAUNCH_MOE(1);
  } else {
    LAUNCH_MOE(2);
  }
#undef LAUNCH_MOE_W
#undef LAUNCH_MOE
  qpn2_check_launch();
}

void sm70_nvfp4_qpn2_moe_gemm(torch::stable::Tensor& out,
                             torch::stable::Tensor& input,
                             torch::stable::Tensor& codes,
                             torch::stable::Tensor& scales,
                             torch::stable::Tensor& global_scale,
                             torch::stable::Tensor& sorted_token_ids,
                             torch::stable::Tensor& expert_ids,
                             torch::stable::Tensor& num_tokens_past_padded,
                             torch::stable::Tensor& topk_weights,
                             int64_t top_k, bool mul_topk_weights,
                             int64_t num_shards, int64_t moe_block_size,
                             bool fuse_act) {
  sm70_nvfp4_qpn2_moe_gemm_impl<false>(out, input, codes, scales,
                                      global_scale, sorted_token_ids,
                                      expert_ids, num_tokens_past_padded,
                                      topk_weights, top_k, mul_topk_weights,
                                      num_shards, moe_block_size, fuse_act);
}

void sm70_nvfp4_qpn2_moe_gemm_fp16b(torch::stable::Tensor& out,
                                    torch::stable::Tensor& input,
                                    torch::stable::Tensor& b_weights,
                                    torch::stable::Tensor& scales_ignored,
                                    torch::stable::Tensor& global_scale,
                                    torch::stable::Tensor& sorted_token_ids,
                                    torch::stable::Tensor& expert_ids,
                                    torch::stable::Tensor&
                                        num_tokens_past_padded,
                                    torch::stable::Tensor& topk_weights,
                                    int64_t top_k, bool mul_topk_weights,
                                    int64_t num_shards,
                                    int64_t moe_block_size, bool fuse_act) {
  sm70_nvfp4_qpn2_moe_gemm_impl<true>(out, input, b_weights, scales_ignored,
                                     global_scale, sorted_token_ids,
                                     expert_ids, num_tokens_past_padded,
                                     topk_weights, top_k, mul_topk_weights,
                                     num_shards, moe_block_size, fuse_act);
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nvfp4_qpn2_ops) {
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_prepare(Tensor weight, Tensor weight_scale) -> Tensor[]");
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_gemm(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, float global_scale, int split_k, int accumulator_chains) "
      "-> ()");
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_prepare_moe(Tensor weight, Tensor weight_scale) -> Tensor[]");
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_moe_gemm(Tensor(a!) out, Tensor input, Tensor codes, "
      "Tensor scales, Tensor global_scale, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_past_padded, "
      "Tensor topk_weights, int top_k, bool mul_topk_weights, "
      "int num_shards, int moe_block_size, bool fuse_act) -> ()");
  sm70_nvfp4_qpn2_ops.def(
      "sm70_nvfp4_qpn2_moe_gemm_fp16b(Tensor(a!) out, Tensor input, "
      "Tensor b_weights, Tensor scales_ignored, Tensor global_scale, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_past_padded, Tensor topk_weights, int top_k, "
      "bool mul_topk_weights, int num_shards, int moe_block_size, "
      "bool fuse_act) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nvfp4_qpn2_ops) {
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_prepare",
                          TORCH_BOX(&sm70_nvfp4_qpn2_prepare));
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_gemm",
                          TORCH_BOX(&sm70_nvfp4_qpn2_gemm));
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_prepare_moe",
                          TORCH_BOX(&sm70_nvfp4_qpn2_prepare_moe));
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_moe_gemm",
                          TORCH_BOX(&sm70_nvfp4_qpn2_moe_gemm));
  sm70_nvfp4_qpn2_ops.impl("sm70_nvfp4_qpn2_moe_gemm_fp16b",
                          TORCH_BOX(&sm70_nvfp4_qpn2_moe_gemm_fp16b));
}
