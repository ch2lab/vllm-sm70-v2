// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// In-tree QSA page4 XQA tc-256-wide decode/prefill kernels + launchers +
// host binding; derived from the flashv100 lineage (pinned be3722bc,
// ownership decision 2026-10-16: in-tree local implementation).

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"
#include "attention/qsa_page4_common.h"
#include "attention/qsa_page4_kv_utils.cuh"
#include "attention/sm70_volta_mma.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>

// The reference references the WMMA wrappers unqualified via
// `volta::` (fused_mma.h, global namespace volta). The in-tree analog is
// sm70_volta_mma.h (namespace sm70gv::volta), which carries the 16x16
// specializations only; the tc-256-wide kernels use 8x32. The 8x32 fragment
// specializations and PTX wrappers below are transcribed verbatim from the
// reference fused_mma.h and added to sm70gv::volta (FLAGGED deviation:
// additive WMMA wrappers, required by the unrenamed kernel code).
namespace sm70gv::volta {

template <>
struct fragment<matrix_a, 8, 32, 16, half, row_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};

template <>
struct fragment<matrix_b, 8, 32, 16, half, col_major> {
  uint32_t x[8];
  static constexpr int num_elements = 16;
};

template <>
struct fragment<accumulator, 8, 32, 16, float> {
  float x[8];
  static constexpr int num_elements = 8;
};

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 8, 32, 16, half, row_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.a.sync.aligned.row.m8n32k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 8, 32, 16, half, col_major>& frag, const half* smem_ptr,
    unsigned ldm) {
  asm volatile(
      "wmma.load.b.sync.aligned.col.m8n32k16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
      : "=r"(frag.x[0]), "=r"(frag.x[1]), "=r"(frag.x[2]), "=r"(frag.x[3]),
        "=r"(frag.x[4]), "=r"(frag.x[5]), "=r"(frag.x[6]), "=r"(frag.x[7])
      : "l"(smem_ptr), "r"(ldm)
      : "memory");
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, 8, 32, 16, float>& frag, const float* smem_ptr,
    unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    asm volatile(
        "wmma.load.c.sync.aligned.row.m8n32k16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
          "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory");
  } else {
    asm volatile(
        "wmma.load.c.sync.aligned.col.m8n32k16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=f"(frag.x[0]), "=f"(frag.x[1]), "=f"(frag.x[2]), "=f"(frag.x[3]),
          "=f"(frag.x[4]), "=f"(frag.x[5]), "=f"(frag.x[6]), "=f"(frag.x[7])
        : "l"(smem_ptr), "r"(ldm)
        : "memory");
  }
}

__device__ __forceinline__ void store_matrix_sync(
    float* smem_ptr, const fragment<accumulator, 8, 32, 16, float>& frag,
    unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    asm volatile(
        "wmma.store.d.sync.aligned.row.m8n32k16.f32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :
        : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]),
          "f"(frag.x[3]), "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]),
          "f"(frag.x[7]), "r"(ldm)
        : "memory");
  } else {
    asm volatile(
        "wmma.store.d.sync.aligned.col.m8n32k16.f32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :
        : "l"(smem_ptr), "f"(frag.x[0]), "f"(frag.x[1]), "f"(frag.x[2]),
          "f"(frag.x[3]), "f"(frag.x[4]), "f"(frag.x[5]), "f"(frag.x[6]),
          "f"(frag.x[7]), "r"(ldm)
        : "memory");
  }
}

__device__ __forceinline__ void mma_sync(
    fragment<accumulator, 8, 32, 16, float>& d,
    const fragment<matrix_a, 8, 32, 16, half, row_major>& a,
    const fragment<matrix_b, 8, 32, 16, half, col_major>& b,
    const fragment<accumulator, 8, 32, 16, float>& c) {
  asm volatile(
      "wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, "
      "{%16,%17,%18,%19,%20,%21,%22,%23}, "
      "{%24,%25,%26,%27,%28,%29,%30,%31};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),
        "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(a.x[4]),
          "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]), "r"(b.x[0]), "r"(b.x[1]),
          "r"(b.x[2]), "r"(b.x[3]), "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]),
          "r"(b.x[7]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));
}

}  // namespace sm70gv::volta

namespace vllm {
namespace qsa_page4 {

namespace volta = sm70gv::volta;

// Reference constants-block member missing from qsa_page4_common.h.
constexpr float kXQANegInf = -1.0e30f;

// Policy stand-in: flash_v100_policy.h is deliberately not ported. The only
// policy field used by this file's launch region is e4m3_scalar_fast, whose
// documented default is enabled; the direct-call environment semantics are
// preserved (VLLM_FLASH_V100_E4M3_SCALAR_FAST, falling back to
// VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST; unset or exactly "1" = enabled).
inline bool e4m3_scalar_fast_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_SCALAR_FAST");
  if (value == nullptr) {
    value = std::getenv("VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST");
  }
  return value == nullptr || (value[0] == '1' && value[1] == '\0');
}

// Stand-in for flash_v100::policy::observation(12) (diagnostic counter).
inline std::atomic<unsigned long long>& decode_e4m3_scalar_fast_observation() {
  static std::atomic<unsigned long long> counter{0};
  return counter;
}

// C10_CUDA_KERNEL_LAUNCH_CHECK equivalent for the stable ABI (the ATen check
// macro is unavailable under torch/csrc/stable).
inline void cuda_launch_check() {
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ",
                  cudaGetErrorString(err));
}

// Two-byte unscaled E5M2 loader used by dot_qk_cache; qsa_page4_kv_utils.cuh
// defines no such loader, so it is transcribed verbatim from the reference
// fp8_kv_utils.cuh (FLAGGED deviation: additive element loader).
__device__ __forceinline__ __half2 load_fp8_e5m2_half2_unscaled(
    const void* __restrict__ cache, const int64_t byte_index) {
  const uint16_t* cache_u16 = reinterpret_cast<const uint16_t*>(cache);
  return fp8_e5m2_pair_to_half2(cache_u16[byte_index >> 1]);
}

int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto" || kv_cache_dtype == "float16" ||
      kv_cache_dtype == "bfloat16") {
    return KV_CACHE_DTYPE_FP16;
  }
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    return KV_CACHE_DTYPE_FP8_E4M3;
  }
  if (kv_cache_dtype == "fp8_e5m2") {
    return KV_CACHE_DTYPE_FP8_E5M2;
  }
  if (kv_cache_dtype == "nvfp4") {
    return KV_CACHE_DTYPE_NVFP4;
  }
  return -1;
}

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kKVStride;
  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    alignas(16) __half k[kXQATCBlockN * kKVStride];
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int) { return reuse_kv.k; }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCQKPipelineSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  static constexpr int kQKStride = kXQATCQKPipelineKVStride;
  struct QKBuffers {
    alignas(16) __half panel[2][kXQATCBlockN * kQKStride];
  };

  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    QKBuffers qk;
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];

  __device__ __forceinline__ __half* k_buffer(int index) {
    return reuse_kv.qk.panel[index];
  }
  __device__ __forceinline__ __half* v_buffer() { return reuse_kv.v; }
};

template <bool PADDED_SMEM>
struct alignas(256) XQATCStagedPVSmem256Wide {
  static constexpr int kKVStride =
      PADDED_SMEM ? kXQATC256WidePaddedKVStride : kXQATCStride;
  alignas(16) __half v[kXQATCStagedPVTileRows * kKVStride];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];
};

template <int SEQ_LEN_ROUTE>
__device__ __forceinline__ bool xqa_seq_len_route_active(
    const int seq_len, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  if constexpr (SEQ_LEN_ROUTE == kXQARouteShortSeqLens) {
    return seq_len < route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024Sawtooth) {
    return (seq_len >= route_seq_len_begin && seq_len < route_seq_len_end) ||
           seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP256Sawtooth) {
    return seq_len < route_seq_len_begin ||
           (seq_len >= route_seq_len_end && seq_len < route_seq_len_final);
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothMid) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteP1024SawtoothFinal) {
    return seq_len >= route_seq_len_final;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteRangeSeqLens) {
    return seq_len >= route_seq_len_begin && seq_len < route_seq_len_end;
  } else if constexpr (SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens) {
    return seq_len >= route_seq_len_begin;
  }
  return true;
}

__device__ __forceinline__ int xqa_sawtooth_partition_size(
    const int seq_len, const int p1024_mid_seq_len, const int p256_long_seq_len,
    const int p1024_final_seq_len) {
  const bool use_p1024 =
      (seq_len >= p1024_mid_seq_len && seq_len < p256_long_seq_len) ||
      seq_len >= p1024_final_seq_len;
  return use_p1024 ? 1024 : 256;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : 0.f;
  if (warp == 0) {
    val = warp_reduce_sum(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_max(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : -1.0e20f;
  if (warp == 0) {
    val = warp_reduce_max(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          int KV_DTYPE = KV_CACHE_DTYPE_FP16, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ uint4 load_xqa_tc_kv_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const uint16_t* __restrict__ e4m3_lut = nullptr,
    const uint16_t* __restrict__ nvfp4_lut = nullptr) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_NVFP4) {
    return load_nvfp4_pool_vector<BLOCK_SIZE, true>(
        kv_cache, page_ids, vec_col, token_offset, panel_offset, block_stride,
        block_size / 4, nvfp4_lut);
  }
  static_assert(BLOCK_SIZE == 0 || BLOCK_SIZE == 4 || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 784 || BLOCK_SIZE == 800 ||
                    BLOCK_SIZE == 1568 || BLOCK_SIZE == 1648 ||
                    BLOCK_SIZE == 3296,
                "Unsupported paged-KV block-size specialization");
  static_assert(!CONTIGUOUS_HKV1_LAYOUT || BLOCK_SIZE == 16 ||
                    BLOCK_SIZE == 800 || BLOCK_SIZE == 1568 ||
                    BLOCK_SIZE == 1648 || BLOCK_SIZE == 3296,
                "The fixed-stride Hkv=1 layout requires a specialized page");
  int logical_block;
  int block_offset;
  if constexpr (BLOCK_SIZE == 4) {
    logical_block = token_offset >> 2;
    block_offset = token_offset & 3;
  } else if constexpr (BLOCK_SIZE == 16) {
    logical_block = token_offset >> 4;
    block_offset = token_offset & 15;
  } else if constexpr (BLOCK_SIZE == 784) {
    logical_block = token_offset / 784;
    block_offset = token_offset - logical_block * 784;
  } else if constexpr (BLOCK_SIZE == 800) {
    // E4M3 long-wave partitions can span four 800-token pages. Keep the
    // constant-page specialization division-free without truncating at page 1.
    logical_block =
        (token_offset >= 800) + (token_offset >= 1600) + (token_offset >= 2400);
    block_offset = token_offset - logical_block * 800;
  } else if constexpr (BLOCK_SIZE == 1568) {
    logical_block = token_offset / 1568;
    block_offset = token_offset - logical_block * 1568;
  } else if constexpr (BLOCK_SIZE == 1648) {
    logical_block = token_offset / 1648;
    block_offset = token_offset - logical_block * 1648;
  } else if constexpr (BLOCK_SIZE == 3296) {
    logical_block = token_offset / 3296;
    block_offset = token_offset - logical_block * 3296;
  } else {
    logical_block = token_offset / block_size;
    block_offset = token_offset % block_size;
  }
  const int physical_block = page_ids[logical_block];
  int64_t physical_offset;
  if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kBlockStride =
        BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
    physical_offset = static_cast<int64_t>(physical_block) * kBlockStride +
                      static_cast<int64_t>(block_offset) * kHeadDim +
                      panel_offset;
  } else {
    physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                      static_cast<int64_t>(block_offset) * token_stride +
                      static_cast<int64_t>(kv_head_idx) * head_stride +
                      panel_offset;
  }
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    const uint4* cache_vec = reinterpret_cast<const uint4*>(kv_cache);
    return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
  } else {
    static_assert(KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2 ||
                      KV_DTYPE == KV_CACHE_DTYPE_NVFP4,
                  "XQA only supports fp16, FP8 E4M3, FP8 E5M2 and NVFP4 KV "
                  "(NVFP4 returns in its own branch above)");
    const uint64_t* cache_vec = reinterpret_cast<const uint64_t*>(kv_cache);
    const uint64_t raw = __ldg(&cache_vec[physical_offset / 8 + vec_col]);
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3) {
      if constexpr (E4M3_SHARED_LUT) {
        return fp8_e4m3fn_vector_to_half8_lut(raw, e4m3_lut);
      } else {
        return fp8_e4m3fn_vector_to_half8(raw);
      }
    } else {
      static_assert(!E4M3_SHARED_LUT,
                    "The E4M3 conversion LUT requires E4M3 KV");
      return fp8_e5m2_vector_to_half8(raw);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = KV_CACHE_DTYPE_FP16, bool FP8_PAIR_LOAD = false,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx = threadIdx.x,
    const uint16_t* __restrict__ e4m3_lut = nullptr,
    const uint16_t* __restrict__ nvfp4_lut = nullptr) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  if constexpr (FP8_PAIR_LOAD) {
    static_assert(KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2,
                  "Paired XQA loads require FP8 KV");
    static_assert(!E4M3_SHARED_LUT,
                  "Paired E4M3 conversion does not use the shared LUT");
    const int pair_stride = panel_d_stride_uint4 / 2;
    const int pair_count = valid_kv_tile_rows * pair_stride;
    for (int pair_idx = copy_thread_idx; pair_idx < pair_count;
         pair_idx += NUM_THREADS) {
      const int row = pair_idx / pair_stride;
      const int vec_pair = pair_idx % pair_stride;
      const int token_offset = tile_page_offset + kv_tile_start + row;
      int logical_block;
      int block_offset;
      if constexpr (BLOCK_SIZE == 4) {
        logical_block = token_offset >> 2;
        block_offset = token_offset & 3;
      } else if constexpr (BLOCK_SIZE == 16) {
        logical_block = token_offset >> 4;
        block_offset = token_offset & 15;
      } else if constexpr (BLOCK_SIZE == 784) {
        logical_block = token_offset / 784;
        block_offset = token_offset - logical_block * 784;
      } else if constexpr (BLOCK_SIZE == 800) {
        logical_block = (token_offset >= 800) + (token_offset >= 1600) +
                        (token_offset >= 2400);
        block_offset = token_offset - logical_block * 800;
      } else if constexpr (BLOCK_SIZE == 1568) {
        logical_block = token_offset / 1568;
        block_offset = token_offset - logical_block * 1568;
      } else if constexpr (BLOCK_SIZE == 1648) {
        logical_block = token_offset / 1648;
        block_offset = token_offset - logical_block * 1648;
      } else if constexpr (BLOCK_SIZE == 3296) {
        logical_block = token_offset / 3296;
        block_offset = token_offset - logical_block * 3296;
      } else {
        logical_block = token_offset / block_size;
        block_offset = token_offset % block_size;
      }
      const int physical_block = page_ids[logical_block];
      int64_t physical_offset;
      if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
        constexpr int64_t kHeadDim = 256;
        constexpr int64_t kPhysicalBlockStride =
            BLOCK_SIZE == 16 ? 16 * kHeadDim : 2 * BLOCK_SIZE * kHeadDim;
        physical_offset =
            static_cast<int64_t>(physical_block) * kPhysicalBlockStride +
            static_cast<int64_t>(block_offset) * kHeadDim + panel_offset;
      } else {
        physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                          static_cast<int64_t>(block_offset) * token_stride +
                          static_cast<int64_t>(kv_head_idx) * head_stride +
                          panel_offset;
      }
      const uint4 raw = __ldg(reinterpret_cast<const uint4*>(kv_cache) +
                              physical_offset / 16 + vec_pair);
      const int shared_offset = row * kv_smem_stride_uint4 + vec_pair * 2;
      const uint64_t raw_lo =
          static_cast<uint64_t>(raw.x) | (static_cast<uint64_t>(raw.y) << 32);
      shared_vec[shared_offset] =
          KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_lo)
              : fp8_e5m2_vector_to_half8(raw_lo);
      const uint64_t raw_hi =
          static_cast<uint64_t>(raw.z) | (static_cast<uint64_t>(raw.w) << 32);
      shared_vec[shared_offset + 1] =
          KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_hi)
              : fp8_e5m2_vector_to_half8(raw_hi);
    }
  } else {
    const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
    for (int copy_idx = copy_thread_idx; copy_idx < copy_count;
         copy_idx += NUM_THREADS) {
      const int row = copy_idx / panel_d_stride_uint4;
      const int vec_col = copy_idx % panel_d_stride_uint4;
      shared_vec[row * kv_smem_stride_uint4 + vec_col] =
          load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                E4M3_SHARED_LUT>(
              kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
              tile_page_offset, kv_tile_start, block_size, kv_head_idx,
              block_stride, token_stride, head_stride, panel_offset, e4m3_lut,
              nvfp4_lut);
    }
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = KV_CACHE_DTYPE_FP16, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel_and_zero(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int copy_thread_idx,
    const uint16_t* __restrict__ e4m3_lut = nullptr,
    const uint16_t* __restrict__ nvfp4_lut = nullptr) {
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  constexpr int kLoadStages = 4;
  for (int copy_base = copy_thread_idx; copy_base < copy_count;
       copy_base += NUM_THREADS * kLoadStages) {
    uint4 staged[kLoadStages];
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        staged[stage] =
            load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE,
                                  E4M3_SHARED_LUT>(
                kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
                tile_page_offset, kv_tile_start, block_size, kv_head_idx,
                block_stride, token_stride, head_stride, panel_offset,
                e4m3_lut, nvfp4_lut);
      }
    }
#pragma unroll
    for (int stage = 0; stage < kLoadStages; ++stage) {
      const int copy_idx = copy_base + stage * NUM_THREADS;
      if (copy_idx < copy_count) {
        const int row = copy_idx / panel_d_stride_uint4;
        const int vec_col = copy_idx % panel_d_stride_uint4;
        shared_vec[row * kv_smem_stride_uint4 + vec_col] = staged[stage];
      }
    }
  }
  for (int copy_idx = copy_thread_idx + copy_count;
       copy_idx < kXQATCBlockN * panel_d_stride_uint4;
       copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
}

template <int D>
__device__ __forceinline__ float dot_qk_half2(const __half* __restrict__ q_ptr,
                                              const __half* __restrict__ k_ptr,
                                              const int lane) {
  static_assert(D % 2 == 0, "Head dim must be even for half2 dot");
  const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
  const __half2* k_ptr2 = reinterpret_cast<const __half2*>(k_ptr);

  float acc = 0.f;
#pragma unroll
  for (int i = lane; i < D / 2; i += kWarpSize) {
    const float2 qv = __half22float2(q_ptr2[i]);
    const float2 kv = __half22float2(k_ptr2[i]);
    acc = fmaf(qv.x, kv.x, acc);
    acc = fmaf(qv.y, kv.y, acc);
  }
  return warp_reduce_sum(acc);
}

template <int D, int KV_DTYPE, bool E4M3_BITS = false>
__device__ __forceinline__ float dot_qk_cache(const __half* __restrict__ q_ptr,
                                             const void* __restrict__ k_cache,
                                             const int64_t k_index_base,
                                             const int lane) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    const __half* k_ptr =
        reinterpret_cast<const __half*>(k_cache) + k_index_base;
    return dot_qk_half2<D>(q_ptr, k_ptr, lane);
  } else if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2) {
    static_assert(D % 2 == 0, "Head dim must be even for e5m2 half2 dot");
    const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
    float acc = 0.f;
#pragma unroll
    for (int i = lane; i < D / 2; i += kWarpSize) {
      const float2 qv = __half22float2(q_ptr2[i]);
      const __half2 k_h2 =
          load_fp8_e5m2_half2_unscaled(
              k_cache, k_index_base + static_cast<int64_t>(i) * 2);
      const float2 kv = __half22float2(k_h2);
      acc = fmaf(qv.x, kv.x, acc);
      acc = fmaf(qv.y, kv.y, acc);
    }
    return warp_reduce_sum(acc);
  } else {
    float acc = 0.f;
#pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float qv = __half2float(q_ptr[d]);
      const float kv =
          load_kv_cache_float_unscaled<KV_DTYPE, E4M3_BITS>(
              k_cache, k_index_base + d);
      acc = fmaf(qv, kv, acc);
    }
    return warp_reduce_sum(acc);
  }
}

template <int D, int PARTITION_SIZE, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool ANCHORED_SWA = false,
          typename PARTIAL_T = __half, bool TP2_E4M3_FAST = false>
__global__ void flash_attention_decode_partition_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, const int batch_size,
    const int max_num_blocks, const int max_num_partitions,
    const int num_heads_q, const int num_heads_kv, const int block_size,
    const int64_t q_stride0, const int64_t q_stride1,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const int64_t v_block_stride, const int64_t v_token_stride,
    const int64_t v_head_stride, const float softmax_scale, const float k_scale,
    const float v_scale, const int window_size_left,
    const int window_size_right, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final,
    const int* __restrict__ anchor_lens, const int anchored_window) {
  // The anchored decode-window mask variant is only generated for the fp16-KV
  // configuration; every other instantiation keeps its original code path.
  static_assert(!ANCHORED_SWA || KV_DTYPE == KV_CACHE_DTYPE_FP16,
                "ANCHORED_SWA requires an fp16 KV cache");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int query_pos = seq_len - 1;
  const int min_token_idx =
      window_size_left >= 0 ? max(0, query_pos - window_size_left) : 0;
  const int max_token_idx =
      window_size_right >= 0 ? min(seq_len - 1, query_pos + window_size_right)
                             : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);

  // Anchored decode window: keys below the per-request prompt length stay
  // globally visible; later (generated) keys must fall inside the sliding
  // window over generated positions. Keys inside [gap_lo, gap_hi) are masked.
  int gap_lo = seq_len;
  int gap_hi = seq_len;
  if constexpr (ANCHORED_SWA) {
    const int anchor_len = anchor_lens[batch_idx];
    gap_lo = anchor_len;
    gap_hi = max(anchor_len, query_pos - anchored_window + 1);
  }
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = KV_DTYPE == KV_CACHE_DTYPE_FP16
                                ? softmax_scale
                                : softmax_scale * k_scale;

  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride2;
  if (part_start >= part_end) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      tmp_out[tmp_out_base + d] = __float2half(0.f);
    }
    if (threadIdx.x == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = -1.0e20f;
      exp_sums[stats_index] = 0.f;
    }
    return;
  }

  const int part_tokens = part_end - part_start;

  __shared__ __half q_shared[D];
  __shared__ float scores_shared[PARTITION_SIZE];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  const __half* q_ptr = q + static_cast<int64_t>(batch_idx) * q_stride0 +
                        static_cast<int64_t>(head_idx) * q_stride1;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_shared[d] = q_ptr[d];
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = part_start + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] =
        block_table[batch_idx * max_num_blocks + logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }
  __syncthreads();

  float local_max = -1.0e20f;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int physical_block = block_idx_shared[token_local];
    const int block_offset = block_offset_shared[token_local];
    const int64_t k_index =
        static_cast<int64_t>(physical_block) * k_block_stride +
        static_cast<int64_t>(block_offset) * k_token_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;

    float score = dot_qk_cache<D, KV_DTYPE, TP2_E4M3_FAST>(q_shared, k_cache,
                                                           k_index, lane);
    if (lane == 0) {
      if constexpr (ANCHORED_SWA) {
        const int token_idx = part_start + token_local;
        if (token_idx >= gap_lo && token_idx < gap_hi) {
          // Masked gap key: never contributes (nulled gap blocks included).
          scores_shared[token_local] = -1.0e30f;
        } else {
          score *= score_scale;
          scores_shared[token_local] = score;
          local_max = fmaxf(local_max, score);
        }
      } else {
        score *= score_scale;
        scores_shared[token_local] = score;
        local_max = fmaxf(local_max, score);
      }
    }
  }

  const float part_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    if constexpr (ANCHORED_SWA) {
      const int token_idx = part_start + i;
      if (token_idx >= gap_lo && token_idx < gap_hi) {
        // Exact zero weight for masked keys (no reliance on expf underflow).
        scores_shared[i] = 0.f;
        continue;
      }
    }
    const float p = __expf(scores_shared[i] - part_max);
    scores_shared[i] = p;
    local_sum += p;
  }
  const float part_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_part_sum = part_sum > 0.f ? 1.f / part_sum : 0.f;
  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    if constexpr (TP2_E4M3_FAST) {
      // Expose independent loads while retaining the ascending-token FMA chain.
#pragma unroll 8
      for (int i = 0; i < part_tokens; ++i) {
        const int physical_block = block_idx_shared[i];
        const int block_offset = block_offset_shared[i];
        const int64_t v_index =
            static_cast<int64_t>(physical_block) * v_block_stride +
            static_cast<int64_t>(block_offset) * v_token_stride +
            static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
        const float vv =
            load_kv_cache_float_unscaled<KV_DTYPE, true>(v_cache,
                                                         v_index);
        acc = fmaf(scores_shared[i], vv, acc);
      }
    } else {
      for (int i = 0; i < part_tokens; ++i) {
        const int physical_block = block_idx_shared[i];
        const int block_offset = block_offset_shared[i];
        const int64_t v_index =
            static_cast<int64_t>(physical_block) * v_block_stride +
            static_cast<int64_t>(block_offset) * v_token_stride +
            static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
        const float vv = load_kv_cache_float_unscaled<KV_DTYPE>(
            v_cache, v_index);
        acc = fmaf(scores_shared[i], vv, acc);
      }
    }
    const float out_scale = KV_DTYPE == KV_CACHE_DTYPE_FP16
                                ? inv_part_sum
                                : inv_part_sum * v_scale;
    if constexpr (std::is_same_v<PARTIAL_T, float>) {
      tmp_out[tmp_out_base + d] = acc * out_scale;
    } else {
      tmp_out[tmp_out_base + d] = __float2half(acc * out_scale);
    }
  }

  if (threadIdx.x == 0) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
    max_logits[stats_index] = part_max;
    exp_sums[stats_index] = part_sum;
  }
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM, int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          bool ALIGNED_PADDED_SMEM, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          bool E4M3_SHARED_LUT = false, typename PARTIAL_T = __half>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_partition_kernel_256_wide(
        const __half* __restrict__ q, const void* __restrict__ k_cache,
        const void* __restrict__ v_cache, PARTIAL_T* __restrict__ tmp_out,
        float* __restrict__ max_logits, float* __restrict__ exp_sums,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale,
        const float k_scale, const float v_scale, const int route_seq_len_begin,
        const int route_seq_len_end, const int route_seq_len_final) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPVPanelDim = kXQATCStride;
  constexpr int kNumPVPanels = D / kPVPanelDim;
  constexpr int kQKPanelDim =
      QK_SW_PIPELINE ? kXQATCQKPipelinePanelDim : kXQATCStride;
  constexpr int kNumQKPanels = D / kQKPanelDim;
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  // Keep softmax P in fp32 through PV for FP16 and E4M3 KV, avoiding a half
  // round-trip before the existing fp32 FMA accumulation. Preserve E5M2's
  // established half-P numerical contract. This branch is compile-time only.
  constexpr bool kKeepPfp32 = KV_DTYPE != KV_CACHE_DTYPE_FP8_E5M2;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int pv_panel_d_stride_uint4 = kPVPanelDim / 8;
  constexpr int qk_smem_stride_uint4 =
      QK_SW_PIPELINE ? SmemLayout::kQKStride / 8 : kv_smem_stride_uint4;
  constexpr int qk_panel_d_stride_uint4 = kQKPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  constexpr bool kPVHalf2 = CONTIGUOUS_HKV1_LAYOUT &&
                            (BLOCK_SIZE == 800 || BLOCK_SIZE == 1568) &&
                            KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one full softmax/PV warp");
  static_assert(
      !QK_SW_PIPELINE || (GROUP_SIZE == 6 && (NUM_THREADS == 6 * kWarpSize ||
                                              NUM_THREADS == 8 * kWarpSize)),
      "The QK software pipeline requires a six- or eight-warp G6 "
      "kernel");
  static_assert(
      !E4M3_SHARED_LUT || KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3,
      "The shared conversion LUT is only valid for E4M3 KV");
  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    const int partition_size_begin =
        route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin;
    partition_size = seq_len < partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int start_token_idx = partition_idx * partition_size;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + partition_size - 1) / partition_size;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(partition_size, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  uint16_t* e4m3_lut = nullptr;
  if constexpr (E4M3_SHARED_LUT) {
    e4m3_lut = reinterpret_cast<uint16_t*>(smem_raw + sizeof(SmemLayout));
    for (int raw = tid; raw < 256; raw += NUM_THREADS) {
      e4m3_lut[raw] = fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw));
    }
  }
  uint16_t* nvfp4_lut = nullptr;
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_NVFP4) {
    // The first __syncthreads() below publishes the LUT to all threads.
    nvfp4_lut = reinterpret_cast<uint16_t*>(smem_raw + sizeof(SmemLayout));
    nvfp4_lut_build(nvfp4_lut, tid, NUM_THREADS);
  }
  __half* sQ = smem.q;
  __half* sK = smem.k_buffer(0);
  __half* sV = smem.v_buffer();
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  int partition_start_page = 0;
  int partition_page_offset = 0;
  if constexpr (PARTITION_PAGE_IDS) {
    int partition_page_count;
    if constexpr (BLOCK_SIZE == 800) {
      partition_start_page = start_token_idx / 800;
      partition_page_offset = start_token_idx - partition_start_page * 800;
      partition_page_count = (partition_page_offset + part_tokens + 799) / 800;
    } else {
      partition_start_page = start_token_idx / block_size;
      partition_page_offset =
          start_token_idx - partition_start_page * block_size;
      partition_page_count =
          (partition_page_offset + part_tokens + block_size - 1) / block_size;
    }
    for (int idx = tid; idx < partition_page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[partition_start_page + idx]);
    }
    __syncthreads();
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    int start_page = partition_start_page;
    int tile_page_offset;
    int page_count = 0;
    if constexpr (PARTITION_PAGE_IDS) {
      tile_page_offset = partition_page_offset + block_n * kXQATCBlockN;
    } else if constexpr (BLOCK_SIZE == 16) {
      start_page = tile_token_start >> 4;
      tile_page_offset = tile_token_start & 15;
      page_count = (tile_page_offset + valid_k_rows + 15) >> 4;
    } else if constexpr (BLOCK_SIZE == 784) {
      start_page = tile_token_start / 784;
      tile_page_offset = tile_token_start - start_page * 784;
      page_count = (tile_page_offset + valid_k_rows + 783) / 784;
    } else if constexpr (BLOCK_SIZE == 800) {
      start_page = tile_token_start / 800;
      tile_page_offset = tile_token_start - start_page * 800;
      page_count = 1 + (tile_page_offset + valid_k_rows > 800);
    } else if constexpr (BLOCK_SIZE == 1568) {
      start_page = tile_token_start / 1568;
      tile_page_offset = tile_token_start - start_page * 1568;
      page_count = (tile_page_offset + valid_k_rows + 1567) / 1568;
    } else if constexpr (KV_DTYPE == KV_CACHE_DTYPE_NVFP4) {
      // The NVFP4 virtual block table holds 4-token microblock pages (196
      // per 784-token pool block); the runtime block_size (784) is the pool
      // block width, not the page granularity the reader indexes
      // (load_nvfp4_pool_vector uses token_offset >> 2).
      start_page = tile_token_start >> 2;
      tile_page_offset = tile_token_start & 3;
      page_count = (tile_page_offset + valid_k_rows + 3) >> 2;
    } else {
      start_page = tile_token_start / block_size;
      tile_page_offset = tile_token_start - start_page * block_size;
      page_count =
          (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    }

    if constexpr (!PARTITION_PAGE_IDS) {
      for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
        smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
      }
      __syncthreads();
    }

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kXQATCBlockN) {
      const int valid_kv_tile_rows =
          min(kXQATCBlockN, valid_k_rows - kv_tile_start);
      // The QK fragments die before softmax/PV. This permits the compiler to
      // reuse their registers for the memory-latency-sensitive PV phase.
      volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::row_major>
          qk_a_frag;
      volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::col_major>
          qk_b_frag;
      volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
          qk_acc_frag;
      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        volta::fill_fragment(qk_acc_frag, 0.0f);
      }

      if constexpr (QK_SW_PIPELINE) {
        constexpr int kConsumerWarps = kXQATCBlockN / WMMA_N;
        constexpr int kProducerThreads =
            NUM_THREADS - kConsumerWarps * kWarpSize;
        const int producer_tid = tid - kConsumerWarps * kWarpSize;
        if (producer_tid >= 0) {
          load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                        kProducerThreads, KV_DTYPE,
                                        E4M3_SHARED_LUT>(
              smem.k_buffer(0), k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, 0, producer_tid, e4m3_lut,
              nvfp4_lut);
        }
        __syncthreads();

#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          if (producer_tid >= 0 && panel_idx + 1 < kNumQKPanels) {
            load_xqa_tc_kv_panel_and_zero<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                                          kProducerThreads, KV_DTYPE,
                                          E4M3_SHARED_LUT>(
                smem.k_buffer((panel_idx + 1) & 1), k_cache, smem.page_ids,
                valid_kv_tile_rows, qk_panel_d_stride_uint4,
                qk_smem_stride_uint4, tile_page_offset, kv_tile_start,
                block_size, kv_head_idx, k_block_stride, k_token_stride,
                k_head_stride, panel_offset + kQKPanelDim, producer_tid,
                e4m3_lut, nvfp4_lut);
          }
          if (warp_id < kConsumerWarps) {
            const int tile_n = warp_id * WMMA_N;
            const __half* sK_panel = smem.k_buffer(panel_idx & 1);
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag,
                  sK_panel + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      } else {
#pragma unroll
        for (int panel_idx = 0; panel_idx < kNumQKPanels; ++panel_idx) {
          const int panel_offset = panel_idx * kQKPanelDim;
          load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                               KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
              sK, k_cache, smem.page_ids, valid_kv_tile_rows,
              qk_panel_d_stride_uint4, qk_smem_stride_uint4, tile_page_offset,
              kv_tile_start, block_size, kv_head_idx, k_block_stride,
              k_token_stride, k_head_stride, panel_offset, threadIdx.x,
              e4m3_lut, nvfp4_lut);
          for (int idx = tid + valid_kv_tile_rows * qk_panel_d_stride_uint4;
               idx < kXQATCBlockN * qk_panel_d_stride_uint4;
               idx += NUM_THREADS) {
            const int row = idx / qk_panel_d_stride_uint4;
            const int vec_col = idx % qk_panel_d_stride_uint4;
            reinterpret_cast<uint4*>(sK)[row * qk_smem_stride_uint4 + vec_col] =
                make_uint4(0, 0, 0, 0);
          }
          __syncthreads();

          if (warp_id < (kXQATCBlockN / WMMA_N)) {
            const int tile_n = warp_id * WMMA_N;
#pragma unroll
            for (int k_tile = 0; k_tile < (kQKPanelDim / WMMA_K); ++k_tile) {
              const int k_offset = k_tile * WMMA_K;
              volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                      SmemLayout::kQStride);
              volta::load_matrix_sync(
                  qk_b_frag, sK + tile_n * SmemLayout::kQKStride + k_offset,
                  SmemLayout::kQKStride);
              volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
            }
          }
          __syncthreads();
        }
      }

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
        for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
          qk_acc_frag.x[i] *= KV_DTYPE == KV_CACHE_DTYPE_FP16
                                  ? softmax_scale
                                  : softmax_scale * k_scale;
        }
        volta::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                 qk_acc_frag, kXQATCBlockN,
                                 volta::mem_row_major);
      }
      __syncthreads();
    }

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      float* sP_row_f = sS + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        if constexpr (kKeepPfp32) {
          reinterpret_cast<float4*>(sP_row_f)[vec_col] =
              make_float4(e0, e1, e2, e3);
        } else {
          packed_exp0 = __float22half2_rn(make_float2(e0, e1));
          packed_exp1 = __float22half2_rn(make_float2(e2, e3));
        }
      }

#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        if constexpr (kKeepPfp32) {
          sP_row_f[c] = (c < valid_k_rows) ? e : 0.0f;
        } else {
          sP_row_h[c] =
              (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
        }
      }

#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if constexpr (!kKeepPfp32) {
        __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
        if (vec_col < vec_cols) {
          const int base_offset = vec_col * 2;
          sP_half2[base_offset] = packed_exp0;
          sP_half2[base_offset + 1] = packed_exp1;
        }
      }

      if (block_n > 0) {
#pragma unroll
        for (int i = 0; i < kAccumsPerThread; ++i) {
          out_acc[i] *= exp_diff;
        }
      }
    }
    __syncthreads();

    for (int panel_idx = 0; panel_idx < kNumPVPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPVPanelDim;
      for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
           kv_tile_start += kXQATCBlockN) {
        const int valid_kv_tile_rows =
            min(kXQATCBlockN, valid_k_rows - kv_tile_start);
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE, FP8_PAIR_LOAD, E4M3_SHARED_LUT>(
            sV, v_cache, smem.page_ids, valid_kv_tile_rows,
            pv_panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, v_block_stride,
            v_token_stride, v_head_stride, panel_offset, threadIdx.x, e4m3_lut,
            nvfp4_lut);
        for (int idx = tid + valid_kv_tile_rows * pv_panel_d_stride_uint4;
             idx < kXQATCBlockN * pv_panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / pv_panel_d_stride_uint4;
          const int vec_col = idx % pv_panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const __half* sP_row = sP + row * kXQATCBlockN + kv_tile_start;
          const float* sP_row_f = sS + row * kXQATCBlockN + kv_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCBlockN; ++token) {
            if (token >= valid_kv_tile_rows) {
              break;
            }
            float prob;
            if constexpr (kKeepPfp32) {
              prob = sP_row_f[token];
            } else {
              prob = __half2float(sP_row[token]);
            }
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
            if constexpr (kPVHalf2) {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / (2 * kWarpSize));
                   ++d_iter) {
                const int local_d2 = lane_id + d_iter * kWarpSize;
                const float2 value = __half22float2(
                    reinterpret_cast<const __half2*>(sV_row)[local_d2]);
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter * 2;
                out_acc[acc_idx] = fmaf(prob, value.x, out_acc[acc_idx]);
                out_acc[acc_idx + 1] =
                    fmaf(prob, value.y, out_acc[acc_idx + 1]);
              }
            } else {
#pragma unroll
              for (int d_iter = 0; d_iter < (kPVPanelDim / kWarpSize);
                   ++d_iter) {
                const int local_d = lane_id + d_iter * kWarpSize;
                const int acc_idx =
                    panel_idx * (kPVPanelDim / kWarpSize) + d_iter;
                out_acc[acc_idx] =
                    fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    PARTIAL_T* tmp_out_ptr =
        tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
        static_cast<int64_t>(head_idx) * tmp_out_stride1 +
        static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    const float output_scale = KV_DTYPE == KV_CACHE_DTYPE_FP16
                                   ? inv_row_sum
                                   : inv_row_sum * v_scale;
    if constexpr (kPVHalf2) {
#pragma unroll
      for (int acc_idx = 0; acc_idx < kAccumsPerThread; ++acc_idx) {
        constexpr int kAccumsPerPanel = kPVPanelDim / kWarpSize;
        const int panel_idx = acc_idx / kAccumsPerPanel;
        const int panel_acc_idx = acc_idx % kAccumsPerPanel;
        const int d_iter = panel_acc_idx / 2;
        const int component = panel_acc_idx & 1;
        const int d = panel_idx * kPVPanelDim + d_iter * (2 * kWarpSize) +
                      thread_in_row * 2 + component;
        const float value = out_acc[acc_idx] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    } else {
      for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
        const float value = out_acc[d / kWarpSize] * output_scale;
        if constexpr (std::is_same_v<PARTIAL_T, float>) {
          tmp_out_ptr[d] = value;
        } else {
          tmp_out_ptr[d] = __float2half(value);
        }
      }
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}

template <int D, int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          typename PARTIAL_T = __half>
__global__ void flash_attention_decode_reduce_kernel(
    const PARTIAL_T* __restrict__ tmp_out, const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, __half* __restrict__ out,
    const int batch_size, const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_partition_size_begin,
    const int route_seq_len_begin, const int route_seq_len_end,
    const int route_seq_len_final) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  int partition_size;
  if constexpr (PARTITION_SIZE == -1) {
    static_assert(SEQ_LEN_ROUTE == kXQARouteWaveLongSeqLens,
                  "Runtime wave partitions require the wave-long route");
    partition_size = seq_len < route_seq_len_end
                         ? 512
                         : (seq_len < route_seq_len_final ? 896 : 1664);
  } else if constexpr (PARTITION_SIZE == 0) {
    partition_size = seq_len < route_partition_size_begin ? 64 : 256;
  } else {
    partition_size = PARTITION_SIZE;
  }
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  (void)active_num_partitions;

  if (seq_len <= 0 || num_partitions <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[static_cast<int64_t>(batch_idx) * out_stride0 +
          static_cast<int64_t>(head_idx) * out_stride1 + d] = __float2half(0.f);
    }
    return;
  }

  extern __shared__ float shared_mem[];
  float* max_shared = shared_mem;
  float* weight_shared = shared_mem + max_num_partitions;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float m = max_logits[stats_index];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float weight =
        exp_sums[stats_index] * __expf(max_shared[i] - global_max);
    weight_shared[i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncthreads();

  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                           static_cast<int64_t>(head_idx) * out_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < num_partitions; ++i) {
      acc = fmaf(weight_shared[i],
                  static_cast<float>(
                      tmp_out[tmp_out_base +
                              static_cast<int64_t>(i) * tmp_out_stride2 + d]),
                  acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

// This separates only the cross-partition reducer. The stats kernel preserves
// the original block-wide max/sum tree, and each output dimension retains its
// original ascending-partition FMA order.
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide(
        const __half* __restrict__ q, const __half* __restrict__ k_cache,
        __half* __restrict__ probabilities, float* __restrict__ max_logits,
        float* __restrict__ exp_sums, float* __restrict__ online_rescales,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t probability_stride0, const int64_t probability_stride1,
        const int64_t probability_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t online_rescale_stride0,
        const int64_t online_rescale_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const float softmax_scale) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  using SmemLayout = XQATCSmem256WideLayout<PADDED_SMEM>;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  static_assert(GROUP_SIZE == 6,
                "Staged D=256 TC XQA is specialized for q_per_kv=6");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one softmax warp");

  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sQ = smem.q;
  __half* sK = smem.reuse_kv.k;
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float row_rescale_reg = 1.f;

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;

    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                    volta::row_major>
        qk_a_frag;
    volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                    volta::col_major>
        qk_b_frag;
    volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        qk_acc_frag;
    if (warp_id < (kXQATCBlockN / WMMA_N)) {
      volta::fill_fragment(qk_acc_frag, 0.0f);
    }

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int idx = tid; idx < valid_k_rows * panel_d_stride_uint4;
           idx += NUM_THREADS) {
        const int row = idx / panel_d_stride_uint4;
        const int vec_col = idx % panel_d_stride_uint4;
        const int token_offset = tile_page_offset + row;
        const int physical_block = smem.page_ids[token_offset / block_size];
        const int block_offset = token_offset % block_size;
        const int64_t physical_offset_half_elements =
            static_cast<int64_t>(physical_block) * k_block_stride +
            static_cast<int64_t>(block_offset) * k_token_stride +
            static_cast<int64_t>(kv_head_idx) * k_head_stride + panel_offset;
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_cache);
        reinterpret_cast<uint4*>(sK)[row * kv_smem_stride_uint4 + vec_col] =
            __ldg(&k_vec[physical_offset_half_elements / 8 + vec_col]);
      }
      for (int idx = tid + valid_k_rows * panel_d_stride_uint4;
           idx < kXQATCBlockN * panel_d_stride_uint4; idx += NUM_THREADS) {
        reinterpret_cast<uint4*>(sK)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        const int tile_n = warp_id * WMMA_N;
#pragma unroll
        for (int k_tile = 0; k_tile < (kPanelDim / WMMA_K); ++k_tile) {
          const int k_offset = k_tile * WMMA_K;
          volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                  SmemLayout::kQStride);
          volta::load_matrix_sync(
              qk_b_frag, sK + tile_n * SmemLayout::kKVStride + k_offset,
              SmemLayout::kKVStride);
          volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
        }
      }
      __syncthreads();
    }

    if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
      for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
        qk_acc_frag.x[i] *= softmax_scale;
      }
      volta::store_matrix_sync(sS + warp_id * WMMA_N, qk_acc_frag, kXQATCBlockN,
                               volta::mem_row_major);
    }
    __syncthreads();

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        packed_exp0 = __float22half2_rn(make_float2(e0, e1));
        packed_exp1 = __float22half2_rn(make_float2(e2, e3));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        sP_row_h[c] =
            (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);
      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }
      if (block_n > 0) {
        row_rescale_reg = exp_diff;
      }

      __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
      if (vec_col < vec_cols) {
        const int base_offset = vec_col * 2;
        sP_half2[base_offset] = packed_exp0;
        sP_half2[base_offset + 1] = packed_exp1;
      }
    }
    __syncthreads();

    for (int idx = tid; idx < GROUP_SIZE * valid_k_rows; idx += NUM_THREADS) {
      const int row = idx / valid_k_rows;
      const int token = idx % valid_k_rows;
      const int head_idx = q_head_base + row;
      __half* probability_ptr =
          probabilities +
          static_cast<int64_t>(batch_idx) * probability_stride0 +
          static_cast<int64_t>(head_idx) * probability_stride1 +
          static_cast<int64_t>(partition_idx) * probability_stride2 +
          block_n * kXQATCBlockN + token;
      *probability_ptr = sP[row * kXQATCBlockN + token];
    }
    __syncthreads();
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      const int head_idx = q_head_base + row;
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = row_max_reg;
      exp_sums[stats_index] = row_sum_reg;
      online_rescales[static_cast<int64_t>(batch_idx) * online_rescale_stride0 +
                      static_cast<int64_t>(head_idx) * online_rescale_stride1 +
                      partition_idx] = row_rescale_reg;
    }
  }
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide(
        const __half* __restrict__ v_cache, const __half* probabilities,
        __half* tmp_out, const float* __restrict__ exp_sums,
        const float* __restrict__ online_rescales,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t online_rescale_stride0,
        const int64_t online_rescale_stride1, const int64_t v_block_stride,
        const int64_t v_token_stride, const int64_t v_head_stride) {
  constexpr int D = 256;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  constexpr int kAccumsPerThread = D / kWarpSize;
  using SmemLayout = XQATCStagedPVSmem256Wide<PADDED_SMEM>;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  static_assert(GROUP_SIZE == 6,
                "Staged D=256 TC XQA is specialized for q_per_kv=6");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one PV warp");

  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;
  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sV = smem.v;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    if (block_n > 0 && tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      float rescale = 1.f;
      if (thread_in_row == 0) {
        const int head_idx = q_head_base + row;
        rescale = online_rescales[static_cast<int64_t>(batch_idx) *
                                      online_rescale_stride0 +
                                  static_cast<int64_t>(head_idx) *
                                      online_rescale_stride1 +
                                  partition_idx];
      }
      rescale = __shfl_sync(0xffffffffu, rescale, 0, kWarpSize);
#pragma unroll
      for (int i = 0; i < kAccumsPerThread; ++i) {
        out_acc[i] *= rescale;
      }
    }

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int v_tile_start = 0; v_tile_start < valid_k_rows;
           v_tile_start += kXQATCStagedPVTileRows) {
        const int valid_v_rows =
            min(kXQATCStagedPVTileRows, valid_k_rows - v_tile_start);
        for (int idx = tid; idx < valid_v_rows * panel_d_stride_uint4;
             idx += NUM_THREADS) {
          const int row = idx / panel_d_stride_uint4;
          const int vec_col = idx % panel_d_stride_uint4;
          const int token_offset = tile_page_offset + v_tile_start + row;
          const int physical_block = smem.page_ids[token_offset / block_size];
          const int block_offset = token_offset % block_size;
          const int64_t physical_offset_half_elements =
              static_cast<int64_t>(physical_block) * v_block_stride +
              static_cast<int64_t>(block_offset) * v_token_stride +
              static_cast<int64_t>(kv_head_idx) * v_head_stride + panel_offset;
          const uint4* v_vec = reinterpret_cast<const uint4*>(v_cache);
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              __ldg(&v_vec[physical_offset_half_elements / 8 + vec_col]);
        }
        for (int idx = tid + valid_v_rows * panel_d_stride_uint4;
             idx < kXQATCStagedPVTileRows * panel_d_stride_uint4;
             idx += NUM_THREADS) {
          reinterpret_cast<uint4*>(sV)[idx] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const int head_idx = q_head_base + row;
          const __half* probability_ptr =
              probabilities +
              static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
              static_cast<int64_t>(head_idx) * tmp_out_stride1 +
              static_cast<int64_t>(partition_idx) * tmp_out_stride2 +
              block_n * kXQATCBlockN + v_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCStagedPVTileRows; ++token) {
            if (token >= valid_v_rows) {
              break;
            }
            float prob = 0.f;
            if (lane_id == 0) {
              prob = __half2float(probability_ptr[token]);
            }
            prob = __shfl_sync(0xffffffffu, prob, 0, kWarpSize);
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
#pragma unroll
            for (int d_iter = 0; d_iter < (kPanelDim / kWarpSize); ++d_iter) {
              const int local_d = lane_id + d_iter * kWarpSize;
              const int acc_idx = panel_idx * (kPanelDim / kWarpSize) + d_iter;
              out_acc[acc_idx] =
                  fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
            }
          }
        }
        __syncthreads();
      }
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum =
        exp_sums[static_cast<int64_t>(batch_idx) * stats_stride0 +
                 static_cast<int64_t>(head_idx) * stats_stride1 +
                 partition_idx];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    __half* tmp_out_ptr = tmp_out +
                          static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                          static_cast<int64_t>(head_idx) * tmp_out_stride1 +
                          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
      tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * inv_row_sum);
    }
  }
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens>
__global__ void flash_attention_decode_xqa_reduce_stats_kernel(
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t stats_stride0, const int64_t stats_stride1,
    const int route_seq_len_begin, const int route_seq_len_end,
    const int route_seq_len_final) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                        route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    return;
  }

  extern __shared__ float max_shared[];
  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float m = max_logits[stats_base + i];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float weight =
        exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
    max_logits[stats_base + i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (threadIdx.x == 0) {
    // The partition sum is dead after weights have been materialized.
    exp_sums[stats_base] = global_sum;
  }
}

template <int D, int PARTITION_SIZE, int D_TILE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, typename PARTIAL_T = __half>
__global__ void flash_attention_decode_xqa_reduce_output_kernel(
    const PARTIAL_T* __restrict__ tmp_out, const float* __restrict__ weights,
    const float* __restrict__ global_sums, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1, const int route_seq_len_begin,
    const int route_seq_len_end, const int route_seq_len_final) {
  static_assert(D_TILE > 0 && D_TILE <= kWarpSize,
                "Split-reduce dimension tile must fit in one warp");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = blockIdx.z * D_TILE + threadIdx.x;
  if (batch_idx >= batch_size || head_idx >= num_heads_q || d >= D) {
    return;
  }

  const int64_t out_index = static_cast<int64_t>(batch_idx) * out_stride0 +
                            static_cast<int64_t>(head_idx) * out_stride1 + d;
  const int seq_len = seq_lens[batch_idx];
  if (!xqa_seq_len_route_active<SEQ_LEN_ROUTE>(seq_len, route_seq_len_begin,
                                               route_seq_len_end,
                                               route_seq_len_final)) {
    return;
  }
  const int partition_size =
      PARTITION_SIZE == 0
          ? xqa_sawtooth_partition_size(seq_len, route_seq_len_begin,
                                         route_seq_len_end, route_seq_len_final)
          : PARTITION_SIZE;
  const int num_partitions =
      min(max_num_partitions, (seq_len + partition_size - 1) / partition_size);
  if (seq_len <= 0 || num_partitions <= 0) {
    out[out_index] = __float2half(0.f);
    return;
  }

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const float global_sum = global_sums[stats_base];
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;

  float acc = 0.f;
  for (int i = 0; i < num_partitions; ++i) {
    acc = fmaf(weights[stats_base + i],
               static_cast<float>(
                   tmp_out[tmp_out_base +
                           static_cast<int64_t>(i) * tmp_out_stride2 + d]),
               acc);
  }
  out[out_index] = __float2half(acc * inv_global_sum);
}

template <int D, int PARTITION_SIZE, int KV_DTYPE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, typename PARTIAL_T = __half>
void launch_flash_attention_decode_paged(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const float softmax_scale, const int launch_num_partitions,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right, cudaStream_t stream,
    const int route_seq_len_begin = 0, const int route_seq_len_end = 0,
    const int route_seq_len_final = 0, const bool launch_reduce = true,
    const int* anchor_lens = nullptr, const int anchored_window = 0) {
  const int batch_size = q.sizes()[0];
  const int num_heads_q = q.sizes()[1];
  const int num_heads_kv = k_cache.sizes()[2];
  const int block_size = k_cache.sizes()[1];
  const int max_num_blocks = block_table.sizes()[1];
  const int max_num_partitions = launch_num_partitions;

  const dim3 partition_grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 reduce_grid(batch_size, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);

  const bool use_anchored = anchor_lens != nullptr && anchored_window > 0;
  // Second kernel version: the anchored decode-window mask is a separate
  // template instantiation, generated only for the fp16-KV configuration;
  // the non-anchored instantiations stay untouched.
  const auto launch_partition = [&](auto anchored_tag, auto fast_tag) {
    constexpr bool kAnchored = decltype(anchored_tag)::value;
    constexpr bool kFast = decltype(fast_tag)::value;
    flash_attention_decode_partition_kernel<
        D, PARTITION_SIZE, KV_DTYPE, SEQ_LEN_ROUTE, kAnchored, PARTIAL_T, kFast>
        <<<partition_grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(q.const_data_ptr()),
            k_cache.const_data_ptr(), v_cache.const_data_ptr(),
            reinterpret_cast<PARTIAL_T*>(tmp_out.mutable_data_ptr()),
            max_logits.mutable_data_ptr<float>(),
            exp_sums.mutable_data_ptr<float>(),
            block_table.const_data_ptr<int>(),
            seq_lens.const_data_ptr<int>(),
            active_num_partitions.const_data_ptr<int>(), batch_size,
            max_num_blocks, max_num_partitions, num_heads_q, num_heads_kv,
            block_size, q.strides()[0], q.strides()[1], tmp_out.strides()[0],
            tmp_out.strides()[1], tmp_out.strides()[2],
            max_logits.strides()[0], max_logits.strides()[1],
            k_cache.strides()[0], k_cache.strides()[1], k_cache.strides()[2],
            v_cache.strides()[0], v_cache.strides()[1], v_cache.strides()[2],
            softmax_scale, k_scale, v_scale, window_size_left,
            window_size_right, route_seq_len_begin, route_seq_len_end,
            route_seq_len_final, anchor_lens, anchored_window);
  };
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    if (use_anchored) {
      launch_partition(std::true_type{}, std::false_type{});
    } else {
      launch_partition(std::false_type{}, std::false_type{});
    }
  } else {
    STD_TORCH_CHECK(!use_anchored,
                    "anchored decode window requires an fp16 KV cache");
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 && D == 256 &&
                  PARTITION_SIZE == 1024 && std::is_same_v<PARTIAL_T, float>) {
      const bool use_fast = e4m3_scalar_fast_enabled() &&
                            window_size_left == -1 && window_size_right == -1;
      if (use_fast) {
        decode_e4m3_scalar_fast_observation().fetch_add(
            1, std::memory_order_relaxed);
        launch_partition(std::false_type{}, std::true_type{});
      } else {
        launch_partition(std::false_type{}, std::false_type{});
      }
    } else {
      launch_partition(std::false_type{}, std::false_type{});
    }
  }

  if (!launch_reduce) {
    return;
  }

  flash_attention_decode_reduce_kernel<D, PARTITION_SIZE, kXQARouteAllSeqLens,
                                       PARTIAL_T>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
          reinterpret_cast<const PARTIAL_T*>(tmp_out.const_data_ptr()),
          max_logits.const_data_ptr<float>(),
          exp_sums.const_data_ptr<float>(),
          seq_lens.const_data_ptr<int>(),
          active_num_partitions.const_data_ptr<int>(),
          reinterpret_cast<__half*>(out.mutable_data_ptr()),
          batch_size, max_num_partitions, num_heads_q, tmp_out.strides()[0],
          tmp_out.strides()[1], tmp_out.strides()[2],
          max_logits.strides()[0], max_logits.strides()[1],
          out.strides()[0], out.strides()[1], 0, 0, 0, 0);
  cuda_launch_check();
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          typename PARTIAL_T = __half>
void launch_flash_attention_decode_xqa_split_reduce(
    torch::stable::Tensor& out, const torch::stable::Tensor& seq_lens,
    const torch::stable::Tensor& tmp_out, torch::stable::Tensor& max_logits,
    torch::stable::Tensor& exp_sums, const int launch_num_partitions,
    const int dim_tile, cudaStream_t stream, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0) {
  const int batch_size = out.sizes()[0];
  const int num_heads_q = out.sizes()[1];
  const dim3 stats_grid(batch_size, num_heads_q, 1);
  const dim3 stats_block(kThreadsPerBlock);
  const size_t stats_shared_mem =
      static_cast<size_t>(launch_num_partitions) * sizeof(float);
  flash_attention_decode_xqa_reduce_stats_kernel<PARTITION_SIZE, SEQ_LEN_ROUTE>
      <<<stats_grid, stats_block, stats_shared_mem, stream>>>(
          max_logits.mutable_data_ptr<float>(),
          exp_sums.mutable_data_ptr<float>(),
          seq_lens.const_data_ptr<int>(), batch_size, launch_num_partitions,
          num_heads_q, max_logits.strides()[0], max_logits.strides()[1],
          route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  cuda_launch_check();

#define LAUNCH_SPLIT_REDUCE_OUTPUT(D_TILE)                                   \
  do {                                                                       \
    const dim3 output_grid(batch_size, num_heads_q,                          \
                           (256 + D_TILE - 1) / D_TILE);                     \
    const dim3 output_block(D_TILE);                                         \
    flash_attention_decode_xqa_reduce_output_kernel<                         \
        256, PARTITION_SIZE, D_TILE, SEQ_LEN_ROUTE, PARTIAL_T>               \
        <<<output_grid, output_block, 0, stream>>>(                          \
            reinterpret_cast<const PARTIAL_T*>(tmp_out.const_data_ptr()),    \
            max_logits.const_data_ptr<float>(),                              \
            exp_sums.const_data_ptr<float>(),                                \
            seq_lens.const_data_ptr<int>(),                                  \
            reinterpret_cast<__half*>(out.mutable_data_ptr()),       \
            batch_size, launch_num_partitions, num_heads_q,                  \
            tmp_out.strides()[0], tmp_out.strides()[1], tmp_out.strides()[2], \
            max_logits.strides()[0], max_logits.strides()[1],                \
            out.strides()[0], out.strides()[1],                              \
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);    \
  } while (0)

  switch (dim_tile) {
    case 16:
      LAUNCH_SPLIT_REDUCE_OUTPUT(16);
      break;
    case 32:
      LAUNCH_SPLIT_REDUCE_OUTPUT(32);
      break;
    default:
      LAUNCH_SPLIT_REDUCE_OUTPUT(8);
      break;
  }
#undef LAUNCH_SPLIT_REDUCE_OUTPUT
  cuda_launch_check();
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM,
          int NUM_THREADS = kXQATC256WideThreads, int MIN_BLOCKS_PER_SM = 1,
          int BLOCK_SIZE = 0, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool ALIGNED_PADDED_SMEM = false,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens, bool QK_SW_PIPELINE = false,
          bool PARTITION_PAGE_IDS = false, bool FP8_PAIR_LOAD = false,
          int KV_DTYPE_OVERRIDE = -1, bool E4M3_SHARED_LUT = false,
          typename PARTIAL_T = std::conditional_t<
              KV_DTYPE_OVERRIDE == KV_CACHE_DTYPE_FP8_E4M3, float,
              __half>>
void launch_flash_attention_decode_paged_xqa_tc_256_wide(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const float softmax_scale, const float k_scale, const float v_scale,
    const int launch_num_partitions, const bool use_split_reduce,
    const int split_reduce_dim_tile, cudaStream_t stream,
    const int route_seq_len_begin = 0, const int route_seq_len_end = 0,
    const int route_seq_len_final = 0, const bool launch_reduce = true) {
  static_assert(!E4M3_SHARED_LUT ||
                    KV_DTYPE_OVERRIDE == KV_CACHE_DTYPE_FP8_E4M3,
                "The shared conversion LUT requires an E4M3 specialization");
  static_assert(KV_DTYPE_OVERRIDE != KV_CACHE_DTYPE_FP8_E4M3 ||
                    std::is_same_v<PARTIAL_T, float>,
                "E4M3 XQA must preserve partition outputs in FP32");
  const int batch_size = q.sizes()[0];
  const int num_heads_q = q.sizes()[1];
  const int num_heads_kv = k_cache.sizes()[2];
  const int max_num_blocks = block_table.sizes()[1];
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  using SmemLayout = std::conditional_t<
      QK_SW_PIPELINE,
      XQATCQKPipelineSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>,
      XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>>;
  const size_t shared_mem = sizeof(SmemLayout) +
                            (E4M3_SHARED_LUT ? 256 * sizeof(uint16_t) : 0) +
                            (KV_DTYPE_OVERRIDE == KV_CACHE_DTYPE_NVFP4
                                 ? kNvfp4LutBytes
                                 : 0);
#define LAUNCH_XQA_PARTITION(KV_DTYPE)                                         \
  do {                                                                         \
    auto partition_kernel =                                                    \
        (void*)flash_attention_decode_xqa_tc_partition_kernel_256_wide<        \
            PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,              \
            MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,             \
            ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,      \
            PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT, PARTIAL_T>;    \
    cudaFuncSetAttribute(partition_kernel,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         shared_mem);                                          \
    if constexpr (MIN_BLOCKS_PER_SM > 1) {                                     \
      cudaFuncSetAttribute(partition_kernel,                                   \
                           cudaFuncAttributePreferredSharedMemoryCarveout,     \
                           100);                                               \
    }                                                                          \
    flash_attention_decode_xqa_tc_partition_kernel_256_wide<                   \
        PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,                  \
        MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,                 \
        ALIGNED_PADDED_SMEM, KV_DTYPE, SEQ_LEN_ROUTE, QK_SW_PIPELINE,          \
        PARTITION_PAGE_IDS, FP8_PAIR_LOAD, E4M3_SHARED_LUT, PARTIAL_T>         \
        <<<partition_grid, NUM_THREADS, shared_mem, stream>>>(                 \
            reinterpret_cast<const __half*>(q.const_data_ptr()),       \
            k_cache.const_data_ptr(), v_cache.const_data_ptr(),                \
            reinterpret_cast<PARTIAL_T*>(tmp_out.mutable_data_ptr()),          \
            max_logits.mutable_data_ptr<float>(),                              \
            exp_sums.mutable_data_ptr<float>(),                                \
            block_table.const_data_ptr<int>(),                                 \
            seq_lens.const_data_ptr<int>(),                                    \
            active_num_partitions.const_data_ptr<int>(), batch_size,           \
            max_num_blocks, launch_num_partitions, num_heads_q, num_heads_kv,  \
            k_cache.sizes()[1], q.strides()[0], q.strides()[1],                \
            tmp_out.strides()[0], tmp_out.strides()[1],                        \
            tmp_out.strides()[2], max_logits.strides()[0],                     \
            max_logits.strides()[1], k_cache.strides()[0],                     \
            k_cache.strides()[1], k_cache.strides()[2],                        \
            v_cache.strides()[0], v_cache.strides()[1], v_cache.strides()[2],  \
            softmax_scale, k_scale, v_scale, route_seq_len_begin,              \
            route_seq_len_end, route_seq_len_final);                           \
  } while (0)

  if constexpr (KV_DTYPE_OVERRIDE == KV_CACHE_DTYPE_NVFP4) {
    STD_TORCH_CHECK(
        k_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
        "NVFP4 XQA requires the uint8 pool KV");
    LAUNCH_XQA_PARTITION(KV_CACHE_DTYPE_NVFP4);
  } else if constexpr (KV_DTYPE_OVERRIDE == KV_CACHE_DTYPE_FP8_E4M3) {
    static_assert(!(FP8_PAIR_LOAD && E4M3_SHARED_LUT),
                  "Paired E4M3 conversion and the shared LUT are exclusive");
    STD_TORCH_CHECK(k_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
                    "E4M3 XQA requires uint8 KV cache");
    LAUNCH_XQA_PARTITION(KV_CACHE_DTYPE_FP8_E4M3);
  } else if constexpr (FP8_PAIR_LOAD) {
    static_assert(!E4M3_SHARED_LUT, "The E4M3 conversion LUT requires E4M3 KV");
    STD_TORCH_CHECK(k_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
                    "Paired XQA loads require uint8 FP8 KV cache");
    LAUNCH_XQA_PARTITION(KV_CACHE_DTYPE_FP8_E5M2);
  } else if (k_cache.scalar_type() == torch::headeronly::ScalarType::Byte) {
    LAUNCH_XQA_PARTITION(KV_CACHE_DTYPE_FP8_E5M2);
  } else {
    LAUNCH_XQA_PARTITION(KV_CACHE_DTYPE_FP16);
  }
#undef LAUNCH_XQA_PARTITION
  cuda_launch_check();

  if (!launch_reduce) {
    return;
  }

  if constexpr (PARTITION_SIZE == -1) {
    STD_TORCH_CHECK(!use_split_reduce,
                    "Runtime wave partitions do not support split reduction");
  }
  if (use_split_reduce) {
    if constexpr (PARTITION_SIZE != -1) {
      launch_flash_attention_decode_xqa_split_reduce<PARTITION_SIZE,
                                                     SEQ_LEN_ROUTE, PARTIAL_T>(
          out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
          split_reduce_dim_tile, stream);
    }
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, PARTITION_SIZE, SEQ_LEN_ROUTE,
                                         PARTIAL_T>
        <<<reduce_grid, block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const PARTIAL_T*>(tmp_out.const_data_ptr()),
            max_logits.const_data_ptr<float>(),
            exp_sums.const_data_ptr<float>(),
            seq_lens.const_data_ptr<int>(),
            active_num_partitions.const_data_ptr<int>(),
            reinterpret_cast<__half*>(out.mutable_data_ptr()),
            batch_size, launch_num_partitions, num_heads_q,
            tmp_out.strides()[0], tmp_out.strides()[1], tmp_out.strides()[2],
            max_logits.strides()[0], max_logits.strides()[1],
            out.strides()[0], out.strides()[1],
            route_seq_len_end > 0 ? route_seq_len_end : route_seq_len_begin,
            route_seq_len_begin, route_seq_len_end, route_seq_len_final);
  }
  cuda_launch_check();
}

template <int PARTITION_SIZE, int BLOCK_SIZE,
          int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          bool FIXED_INTERLEAVED_HKV1_LAYOUT = false>
void launch_flash_attention_decode_paged_xqa_e4m3_g6_aligned_page(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const float softmax_scale, const float k_scale, const float v_scale,
    const int launch_num_partitions, cudaStream_t stream,
    const int route_seq_len_begin = 0, const int route_seq_len_end = 0,
    const int route_seq_len_final = 0, const bool launch_reduce = true) {
  launch_flash_attention_decode_paged_xqa_tc_256_wide<
      PARTITION_SIZE, 6, true, kXQATCG6DualCtaThreads, 2, BLOCK_SIZE,
      FIXED_INTERLEAVED_HKV1_LAYOUT, false, SEQ_LEN_ROUTE, false, true, false,
      KV_CACHE_DTYPE_FP8_E4M3, true>(
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
      launch_num_partitions, false, 8, stream, route_seq_len_begin,
      route_seq_len_end, route_seq_len_final, launch_reduce);
}

template <int PARTITION_SIZE, int SEQ_LEN_ROUTE = kXQARouteAllSeqLens,
          bool ENABLE_FIXED_INTERLEAVED_HKV1_LAYOUT = false>
void launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const float softmax_scale, const float k_scale, const float v_scale,
    const int launch_num_partitions, cudaStream_t stream,
    const bool fixed_interleaved_hkv1_layout, const int route_seq_len_begin = 0,
    const int route_seq_len_end = 0, const int route_seq_len_final = 0,
    const bool launch_reduce = true) {
#define LAUNCH_E4M3_ALIGNED_PAGE(BLOCK_SIZE, FIXED_LAYOUT)                   \
  launch_flash_attention_decode_paged_xqa_e4m3_g6_aligned_page<              \
      PARTITION_SIZE, BLOCK_SIZE, SEQ_LEN_ROUTE, FIXED_LAYOUT>(              \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,  \
      exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,      \
      launch_num_partitions, stream, route_seq_len_begin, route_seq_len_end, \
      route_seq_len_final, launch_reduce)

  if (k_cache.sizes()[1] == 800) {
    if constexpr (ENABLE_FIXED_INTERLEAVED_HKV1_LAYOUT) {
      if (fixed_interleaved_hkv1_layout) {
        LAUNCH_E4M3_ALIGNED_PAGE(800, true);
      } else {
        LAUNCH_E4M3_ALIGNED_PAGE(800, false);
      }
    } else {
      LAUNCH_E4M3_ALIGNED_PAGE(800, false);
    }
  } else if (k_cache.sizes()[1] == 1568) {
    if constexpr (ENABLE_FIXED_INTERLEAVED_HKV1_LAYOUT) {
      if (fixed_interleaved_hkv1_layout) {
        LAUNCH_E4M3_ALIGNED_PAGE(1568, true);
      } else {
        LAUNCH_E4M3_ALIGNED_PAGE(1568, false);
      }
    } else {
      LAUNCH_E4M3_ALIGNED_PAGE(1568, false);
    }
  } else {
    STD_TORCH_CHECK(!fixed_interleaved_hkv1_layout,
                    "Fixed E4M3 Hkv=1 layout requires page 800 or 1568");
    LAUNCH_E4M3_ALIGNED_PAGE(0, false);
  }
#undef LAUNCH_E4M3_ALIGNED_PAGE
}

void launch_flash_attention_decode_paged_xqa_tc_256_staged(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& online_rescales,
    const torch::stable::Tensor& active_num_partitions,
    const float softmax_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream) {
  constexpr int kGroupSize = 6;
  constexpr int kQKThreads = kXQATCG6DualCtaThreads;
  constexpr int kPVThreads = kXQATCG6DualCtaThreads;
  constexpr int kQKMinBlocksPerSM = 2;
  constexpr int kPVMinBlocksPerSM = 4;
  using QKSmem = XQATCSmem256WideLayout<true>;
  using PVSmem = XQATCStagedPVSmem256Wide<true>;

  const int batch_size = q.sizes()[0];
  const int num_heads_q = q.sizes()[1];
  const int num_heads_kv = k_cache.sizes()[2];
  const int max_num_blocks = block_table.sizes()[1];
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  const size_t qk_shared_mem = sizeof(QKSmem);
  const size_t pv_shared_mem = sizeof(PVSmem);
  auto qk_kernel =
      (void*)flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide<
          256, kGroupSize, true, kQKThreads, kQKMinBlocksPerSM>;
  auto pv_kernel =
      (void*)flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide<
          256, kGroupSize, true, kPVThreads, kPVMinBlocksPerSM>;
  const cudaError_t qk_smem_status = cudaFuncSetAttribute(
      qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, qk_shared_mem);
  STD_TORCH_CHECK(qk_smem_status == cudaSuccess,
                  "Failed to set staged XQA QK shared memory: ",
                  cudaGetErrorString(qk_smem_status));
  const cudaError_t qk_carveout_status = cudaFuncSetAttribute(
      qk_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  STD_TORCH_CHECK(qk_carveout_status == cudaSuccess,
                  "Failed to set staged XQA QK shared-memory carveout: ",
                  cudaGetErrorString(qk_carveout_status));
  const cudaError_t pv_smem_status = cudaFuncSetAttribute(
      pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pv_shared_mem);
  STD_TORCH_CHECK(pv_smem_status == cudaSuccess,
                  "Failed to set staged XQA PV shared memory: ",
                  cudaGetErrorString(pv_smem_status));
  const cudaError_t pv_carveout_status = cudaFuncSetAttribute(
      pv_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  STD_TORCH_CHECK(pv_carveout_status == cudaSuccess,
                  "Failed to set staged XQA PV shared-memory carveout: ",
                  cudaGetErrorString(pv_carveout_status));

  flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide<
      256, kGroupSize, true, kQKThreads, kQKMinBlocksPerSM>
      <<<partition_grid, kQKThreads, qk_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(q.const_data_ptr()),
          reinterpret_cast<const __half*>(k_cache.const_data_ptr()),
          reinterpret_cast<__half*>(tmp_out.mutable_data_ptr()),
          max_logits.mutable_data_ptr<float>(),
          exp_sums.mutable_data_ptr<float>(),
          online_rescales.mutable_data_ptr<float>(),
          block_table.const_data_ptr<int>(),
          seq_lens.const_data_ptr<int>(),
          active_num_partitions.const_data_ptr<int>(), batch_size,
          max_num_blocks, launch_num_partitions, num_heads_q,
          num_heads_kv, k_cache.sizes()[1], q.strides()[0], q.strides()[1],
          tmp_out.strides()[0], tmp_out.strides()[1], tmp_out.strides()[2],
          max_logits.strides()[0], max_logits.strides()[1],
          online_rescales.strides()[0], online_rescales.strides()[1],
          k_cache.strides()[0], k_cache.strides()[1], k_cache.strides()[2],
          softmax_scale);
  cuda_launch_check();

  flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide<
      256, kGroupSize, true, kPVThreads, kPVMinBlocksPerSM>
      <<<partition_grid, kPVThreads, pv_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(v_cache.const_data_ptr()),
          reinterpret_cast<const __half*>(tmp_out.const_data_ptr()),
          reinterpret_cast<__half*>(tmp_out.mutable_data_ptr()),
          exp_sums.const_data_ptr<float>(),
          online_rescales.const_data_ptr<float>(),
          block_table.const_data_ptr<int>(),
          seq_lens.const_data_ptr<int>(),
          active_num_partitions.const_data_ptr<int>(), batch_size,
          max_num_blocks, launch_num_partitions, num_heads_q,
          num_heads_kv, v_cache.sizes()[1], tmp_out.strides()[0],
          tmp_out.strides()[1], tmp_out.strides()[2],
          max_logits.strides()[0], max_logits.strides()[1],
          online_rescales.strides()[0], online_rescales.strides()[1],
          v_cache.strides()[0], v_cache.strides()[1], v_cache.strides()[2]);
  cuda_launch_check();

  if (use_split_reduce) {
    launch_flash_attention_decode_xqa_split_reduce<256>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream);
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 reduce_block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, 256>
        <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const __half*>(tmp_out.const_data_ptr()),
            max_logits.const_data_ptr<float>(),
            exp_sums.const_data_ptr<float>(),
            seq_lens.const_data_ptr<int>(),
            active_num_partitions.const_data_ptr<int>(),
            reinterpret_cast<__half*>(out.mutable_data_ptr()),
            batch_size, launch_num_partitions, num_heads_q,
            tmp_out.strides()[0], tmp_out.strides()[1], tmp_out.strides()[2],
            max_logits.strides()[0], max_logits.strides()[1],
            out.strides()[0], out.strides()[1], 0, 0, 0, 0);
  }
  cuda_launch_check();
}

torch::stable::Tensor flash_attention_decode_paged(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    std::optional<torch::stable::Tensor> out_,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right,
    std::optional<torch::stable::Tensor> anchor_lens,
    const int64_t anchored_window) {
  STD_TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  STD_TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
                  "k/v cache must be on CUDA");
  STD_TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                  "block_table and seq_lens must be on CUDA");
  STD_TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() &&
                      exp_sums.is_cuda(),
                  "workspace tensors must be on CUDA");
  STD_TORCH_CHECK(active_num_partitions.is_cuda(),
                  "active_num_partitions must be on CUDA");
  STD_TORCH_CHECK(
      q.scalar_type() == torch::headeronly::ScalarType::Half, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  STD_TORCH_CHECK(kv_dtype_code >= 0,
                  "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == KV_CACHE_DTYPE_FP16) {
    STD_TORCH_CHECK(
        k_cache.scalar_type() == torch::headeronly::ScalarType::Half,
        "k_cache must be fp16");
    STD_TORCH_CHECK(
        v_cache.scalar_type() == torch::headeronly::ScalarType::Half,
        "v_cache must be fp16");
  } else {
    STD_TORCH_CHECK(
        k_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
        "fp8 k_cache must be stored as uint8");
    STD_TORCH_CHECK(
        v_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
        "fp8 v_cache must be stored as uint8");
    STD_TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                    "fp8 k/v scales must be positive");
  }
  STD_TORCH_CHECK(
      tmp_out.scalar_type() == torch::headeronly::ScalarType::Half ||
          (kv_dtype_code == KV_CACHE_DTYPE_FP8_E4M3 &&
           tmp_out.scalar_type() == torch::headeronly::ScalarType::Float),
      "tmp_out must be fp16, or fp32 for E4M3 scalar decode");
  STD_TORCH_CHECK(max_logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "max_logits must be fp32");
  STD_TORCH_CHECK(exp_sums.scalar_type() == torch::headeronly::ScalarType::Float,
                  "exp_sums must be fp32");
  STD_TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int,
      "block_table must be int32");
  STD_TORCH_CHECK(
      seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
      "seq_lens must be int32");
  STD_TORCH_CHECK(
      active_num_partitions.scalar_type() == torch::headeronly::ScalarType::Int,
      "active_num_partitions must be int32");
  STD_TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  STD_TORCH_CHECK(k_cache.dim() == 4,
                  "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  STD_TORCH_CHECK(v_cache.dim() == 4,
                  "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  STD_TORCH_CHECK(
      block_table.dim() == 2,
      "block_table must have shape [B, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  STD_TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  STD_TORCH_CHECK(max_logits.dim() == 3,
                  "max_logits must have shape [B_cap, H, P]");
  STD_TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  STD_TORCH_CHECK(
      active_num_partitions.dim() == 1 &&
          active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  STD_TORCH_CHECK(q.strides()[q.dim() - 1] == 1,
                  "q last dim must be contiguous");
  STD_TORCH_CHECK(k_cache.strides()[k_cache.dim() - 1] == 1,
                  "k_cache last dim must be contiguous");
  STD_TORCH_CHECK(v_cache.strides()[v_cache.dim() - 1] == 1,
                  "v_cache last dim must be contiguous");
  STD_TORCH_CHECK(tmp_out.strides()[tmp_out.dim() - 1] == 1,
                  "tmp_out last dim must be contiguous");

  const int batch_size = q.sizes()[0];
  const int num_heads_q = q.sizes()[1];
  const int head_dim = q.sizes()[2];
  const int num_heads_kv = k_cache.sizes()[2];

  STD_TORCH_CHECK(q.sizes()[0] <= block_table.sizes()[0],
                  "block_table batch size must cover q batch size");
  STD_TORCH_CHECK(q.sizes()[0] <= seq_lens.sizes()[0],
                  "seq_lens batch size must cover q batch size");
  STD_TORCH_CHECK(q.sizes()[0] <= tmp_out.sizes()[0],
                  "tmp_out batch size must cover q batch size");
  STD_TORCH_CHECK(num_heads_q == tmp_out.sizes()[1],
                  "tmp_out head dimension mismatch");
  STD_TORCH_CHECK(head_dim == tmp_out.sizes()[3], "tmp_out head_dim mismatch");
  STD_TORCH_CHECK(max_logits.sizes()[0] == tmp_out.sizes()[0] &&
                      max_logits.sizes()[1] == tmp_out.sizes()[1] &&
                      max_logits.sizes()[2] == tmp_out.sizes()[2],
                  "max_logits shape mismatch");
  STD_TORCH_CHECK(exp_sums.sizes().equals(max_logits.sizes()),
                  "exp_sums shape mismatch");
  STD_TORCH_CHECK(num_heads_q % num_heads_kv == 0,
                  "num_heads_q must be divisible by num_heads_kv");
  STD_TORCH_CHECK(k_cache.sizes()[3] == head_dim,
                  "k_cache head_dim mismatch");
  STD_TORCH_CHECK(v_cache.sizes()[3] == head_dim,
                  "v_cache head_dim mismatch");
  STD_TORCH_CHECK(
      partition_size == 256 || partition_size == 512 ||
          partition_size == 1024,
      "Unsupported decode partition_size: ", partition_size);
  STD_TORCH_CHECK(
      launch_num_partitions > 0 &&
          launch_num_partitions <= tmp_out.sizes()[2],
      "launch_num_partitions must be in (0, tmp_out.size(2)]");
  STD_TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
                  "window sizes must be >= -1");

  STD_TORCH_CHECK(anchored_window >= 0 && anchored_window <= INT_MAX,
                  "anchored_window must fit in a non-negative int32");
  STD_TORCH_CHECK((anchored_window > 0) == anchor_lens.has_value(),
                  "anchor_lens and a positive anchored_window must be provided "
                  "together");
  const bool use_anchored = anchor_lens.has_value();
  const int* anchor_lens_ptr = nullptr;
  if (use_anchored) {
    const torch::stable::Tensor& anchors = anchor_lens.value();
    STD_TORCH_CHECK(anchors.is_cuda(), "anchor_lens must be on CUDA");
    STD_TORCH_CHECK(anchors.device() == q.device(),
                    "anchor_lens must be on the q device");
    STD_TORCH_CHECK(
        anchors.scalar_type() == torch::headeronly::ScalarType::Int,
        "anchor_lens must be int32");
    STD_TORCH_CHECK(anchors.dim() == 1 && anchors.sizes()[0] == batch_size,
                    "anchor_lens must have shape [B]");
    STD_TORCH_CHECK(anchors.is_contiguous(),
                    "anchor_lens must be contiguous");
    STD_TORCH_CHECK(kv_dtype_code == KV_CACHE_DTYPE_FP16,
                    "anchored decode window requires an fp16 KV cache");
    STD_TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                    "anchored decode window cannot be combined with "
                    "sliding-window attention");
    anchor_lens_ptr = anchors.const_data_ptr<int>();
  }

  torch::stable::Tensor out =
      out_.has_value() ? out_.value() : torch::stable::empty_like(q);
  STD_TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::Half,
      "out must be fp16");
  STD_TORCH_CHECK(out.sizes().equals(q.sizes()),
                  "out must have same shape as q");
  STD_TORCH_CHECK(out.strides()[out.dim() - 1] == 1,
                  "out last dim must be contiguous");

  auto stream = get_current_cuda_stream(q.get_device_index());
  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

#define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                          \
  launch_flash_attention_decode_paged<HDIM, PARTITION, KV_DTYPE_CODE>(        \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,   \
      exp_sums, active_num_partitions, softmax_scale, launch_num_partitions,  \
      k_scale, v_scale, window_size_left, window_size_right, stream, 0, 0, 0, \
      true, anchor_lens_ptr, static_cast<int>(anchored_window))

#define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                   \
  do {                                                                        \
    switch (kv_dtype_code) {                                                  \
      case KV_CACHE_DTYPE_FP16:                                               \
        LAUNCH_TYPED(HDIM, PARTITION, KV_CACHE_DTYPE_FP16);                   \
        break;                                                               \
      case KV_CACHE_DTYPE_FP8_E4M3:                                            \
        if (tmp_out.scalar_type() == torch::headeronly::ScalarType::Float) {  \
          launch_flash_attention_decode_paged<                               \
              HDIM, PARTITION, KV_CACHE_DTYPE_FP8_E4M3,                       \
              kXQARouteAllSeqLens, float>(                                    \
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,       \
              max_logits, exp_sums, active_num_partitions, softmax_scale,     \
              launch_num_partitions, k_scale, v_scale, window_size_left,      \
              window_size_right, stream);                                     \
        } else {                                                              \
          LAUNCH_TYPED(HDIM, PARTITION, KV_CACHE_DTYPE_FP8_E4M3);              \
        }                                                                     \
        break;                                                                \
      case KV_CACHE_DTYPE_FP8_E5M2:                                            \
        LAUNCH_TYPED(HDIM, PARTITION, KV_CACHE_DTYPE_FP8_E5M2);                \
        break;                                                                \
      default:                                                                \
        STD_TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
    }                                                                         \
  } while (0)

#define LAUNCH_BY_PARTITION(HDIM)                                           \
  do {                                                                      \
    switch (partition_size) {                                               \
      case 256:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 256);                                      \
        break;                                                              \
      case 512:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 512);                                      \
        break;                                                              \
      case 1024:                                                            \
        LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                     \
        break;                                                              \
      default:                                                              \
        STD_TORCH_CHECK(                                                    \
            false, "Unsupported decode partition_size: ", partition_size);  \
    }                                                                       \
  } while (0)

  switch (head_dim) {
    case 64:
      LAUNCH_BY_PARTITION(64);
      break;
    case 80:
      LAUNCH_BY_PARTITION(80);
      break;
    case 96:
      LAUNCH_BY_PARTITION(96);
      break;
    case 112:
      LAUNCH_BY_PARTITION(112);
      break;
    case 128:
      LAUNCH_BY_PARTITION(128);
      break;
    case 256:
      LAUNCH_BY_PARTITION(256);
      break;
    default:
      STD_TORCH_CHECK(false,
                      "Unsupported head_dim for paged decode: ", head_dim);
  }

#undef LAUNCH_BY_PARTITION
#undef LAUNCH_BY_KV_DTYPE
#undef LAUNCH_TYPED

  cuda_launch_check();
  return out;
}

}  // namespace qsa_page4
}  // namespace vllm
