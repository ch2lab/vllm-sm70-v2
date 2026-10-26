// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// In-tree QSA page4 grouped sparse verify kernel; derived from the flashv100
// lineage (pinned be3722bc, ownership decision 2026-10-16: in-tree local
// implementation). Transcribes the grouped sparse page4 verify entry point
// (flash_attention_grouped_verify_e5m2_partial_kernel with SPARSE_PAGE4) and
// the device types/helpers it requires.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "attention/qsa_page4_common.h"
#include "attention/qsa_page4_kv_utils.cuh"
#include "attention/sm70_volta_mma.h"
#include "dispatch_utils.h"
#include "torch_utils.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace vllm {
namespace qsa_page4 {

// The transcribed device code targets the reference's WMMA layer
// (include/fused_mma.h, namespace "volta") and its flash_v100 dtype namespace.
// The in-tree WMMA analog is sm70gv::volta; the KV dtype enums (KV_CACHE_DTYPE_*)
// live in this namespace. These aliases keep the transcribed bodies bit-exact.
namespace volta = sm70gv::volta;
namespace flash_v100 = qsa_page4;

// ported from flash_decode_paged.cu:70
constexpr float kXQANegInf = -1.0e30f;

// Output-tiling constants that qsa_page4_common.h does not define (the other
// kGroupedVerify* geometry constants come from that header).
// ported from flash_decode_paged.cu:1861-1864
constexpr int kGroupedVerifyOutputTiles =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyHeadDim / 16);
constexpr int kGroupedVerifyOutputTilesPerWarp =
    kGroupedVerifyOutputTiles / kGroupedVerifyWarps;

// warp reduces (ported from flash_decode_paged.cu:567-581)
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

// load_xqa_tc_kv_vector (ported from flash_decode_paged.cu:734-822)
template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ uint4 load_xqa_tc_kv_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int pages_per_block,
    const uint16_t* __restrict__ e4m3_lut = nullptr,
    const uint16_t* __restrict__ nvfp4_lut = nullptr,
    const int abl = 0) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4) {
    return load_nvfp4_pool_vector<BLOCK_SIZE, true>(
        kv_cache, page_ids, vec_col, token_offset, panel_offset, block_stride,
        pages_per_block, nvfp4_lut);
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
  const int physical_block =
      (abl == 3) ? 0 : page_ids[logical_block];  // SAFE: physical block 0 (identity mapping read OOB pages and crashed the engine, incident #3)
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
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const uint4* cache_vec = reinterpret_cast<const uint4*>(kv_cache);
    return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
  } else {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4,
                  "XQA only supports fp16, FP8 E4M3, FP8 E5M2 and NVFP4 KV "
                  "(NVFP4 returns in its own branch above)");
    const uint64_t* cache_vec = reinterpret_cast<const uint64_t*>(kv_cache);
    const uint64_t raw = __ldg(&cache_vec[physical_offset / 8 + vec_col]);
    if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3) {
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

// load_xqa_tc_kv_panel (ported from flash_decode_paged.cu:824-923)
template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16,
          bool FP8_PAIR_LOAD = false, bool E4M3_SHARED_LUT = false>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset, const int pages_per_block,
    const int copy_thread_idx = threadIdx.x,
    const uint16_t* __restrict__ e4m3_lut = nullptr,
    const uint16_t* __restrict__ nvfp4_lut = nullptr,
    // ABL (offline diagnostic, experiment builds only; 0 = production):
    //   3 = replace the page-table indirection with the identity mapping,
    //       isolating the cost of the SCATTERED page gathers.
    const int abl = 0) {
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  if constexpr (FP8_PAIR_LOAD) {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 ||
                      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
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
      const int physical_block =
          (abl == 3) ? 0 : page_ids[logical_block];  // SAFE: physical block 0 (identity mapping read OOB pages and crashed the engine, incident #3)
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
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_lo)
              : fp8_e5m2_vector_to_half8(raw_lo);
      const uint64_t raw_hi =
          static_cast<uint64_t>(raw.z) | (static_cast<uint64_t>(raw.w) << 32);
      shared_vec[shared_offset + 1] =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3
              ? fp8_e4m3fn_vector_to_half8_fast(raw_hi)
              : fp8_e5m2_vector_to_half8(raw_hi);
    }
  } else {
    if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4) {
      // 16-dim pairs: one 8B data load + one scale byte per pair.
      const int pair_stride = panel_d_stride_uint4 / 2;
      const int pair_count = valid_kv_tile_rows * pair_stride;
      for (int pair_idx = copy_thread_idx; pair_idx < pair_count;
           pair_idx += NUM_THREADS) {
        const int row = pair_idx / pair_stride;
        const int vec_pair = pair_idx % pair_stride;
        const int token_offset = tile_page_offset + kv_tile_start + row;
        uint4 lo, hi;
        load_nvfp4_pool_vector_pair<BLOCK_SIZE, true>(
            kv_cache, page_ids, vec_pair, token_offset, panel_offset,
            block_stride, pages_per_block, lo, hi, nvfp4_lut);
        shared_vec[row * kv_smem_stride_uint4 + 2 * vec_pair] = lo;
        shared_vec[row * kv_smem_stride_uint4 + 2 * vec_pair + 1] = hi;
      }
      return;
    }
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
              block_stride, token_stride, head_stride, panel_offset,
              pages_per_block, e4m3_lut, nvfp4_lut, abl);
    }
  }
}

// Grouped-verify support types + device helpers (ported from flash_decode_paged.cu:1866-2035; the kGroupedVerify* geometry constants come from qsa_page4_common.h)
template <int MAX_QUERY_TOKENS>
struct GroupedVerifyTraits {
  static_assert(MAX_QUERY_TOKENS == kGroupedVerifyQ8MaxQ ||
                    MAX_QUERY_TOKENS == kGroupedVerifyQ16MaxQ,
                "grouped verifier supports q8 and q16 workspaces only");
  static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
  static constexpr int kHeadGroups = kGroupedVerifyHeads / kHeadsPerCta;
  static constexpr int kSplits = kGroupedVerifyWorkspaceRows / MAX_QUERY_TOKENS;
  static_assert(MAX_QUERY_TOKENS * kHeadsPerCta == kGroupedVerifyRows,
                "grouped verifier CTA must retain 48 rows");
  static_assert(kGroupedVerifyHeads % kHeadsPerCta == 0,
                "query heads must divide evenly across grouped CTAs");
  static_assert(kSplits * MAX_QUERY_TOKENS == kGroupedVerifyWorkspaceRows,
                "q8 and q16 workspaces must have equal element counts");
};

struct alignas(256) GroupedVerifySmem {
  union {
    struct {
      alignas(16) __half q[kGroupedVerifyRows * kGroupedVerifyQStride];
      alignas(16) __half kv[kGroupedVerifyBlockN * kGroupedVerifyKVStride];
      alignas(16) float scores[kGroupedVerifyRows * kGroupedVerifyScoreStride];
      alignas(16) __half probs[kGroupedVerifyRows * kGroupedVerifyProbStride];
    } compute;
    alignas(16) float output[kGroupedVerifyRows * kGroupedVerifyHeadDim];
  } storage;
  alignas(16) float row_max[kGroupedVerifyRows];
  alignas(16) float row_sum[kGroupedVerifyRows];
  alignas(16) float row_scale[kGroupedVerifyRows];
  alignas(16) int page_ids[kGroupedVerifyPageIdsCapacity];
  alignas(16) uint32_t sparse_token_masks[kGroupedVerifyBlockN / 4];
  // NVFP4 dequant LUTs (built once per CTA by the NVFP4 specialization;
  // dead storage otherwise).
  alignas(16) uint16_t nvfp4_lut[kNvfp4LutBytes / 2];
};

static_assert(sizeof(GroupedVerifySmem) <= 64 * 1024,
              "packed grouped verifier must fit Volta's 64 KiB opt-in budget");
static_assert(kGroupedVerifyOutputTiles % kGroupedVerifyWarps == 0,
              "output tiles must divide evenly across warps");

template <int MAX_QUERY_TOKENS, bool SINGLE_QUERY>
__device__ __forceinline__ int grouped_verify_active_splits(
    const int total_kv) {
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  constexpr int kMinTokensPerSplit =
      SINGLE_QUERY ? kGroupedVerifySingleQueryMinTokensPerSplit
                   : kGroupedVerifyMinTokensPerSplit;
  const int active_splits =
      min(Traits::kSplits,
          max(1, (total_kv + kMinTokensPerSplit - 1) / kMinTokensPerSplit));
  if constexpr (SINGLE_QUERY) {
    return active_splits;
  }
  return total_kv <= kGroupedVerifyShortContextMaxTokens ? 1 : active_splits;
}

template <bool COMPENSATE = false>
__device__ __forceinline__ void grouped_verify_qk(
    const __half* __restrict__ shared_q, const __half* __restrict__ shared_k,
    float* __restrict__ shared_scores, const float qk_scale,
    const int active_m_tiles) {
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kGroupedVerifyQKWarps) {
    return;
  }

  const int m_tile = warp_id / (kGroupedVerifyBlockN / 16);
  if ((active_m_tiles & (1 << m_tile)) == 0) {
    return;
  }
  const int n_tile = warp_id % (kGroupedVerifyBlockN / 16);
  volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
      q_fragment;
  volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::col_major>
      k_fragment;
  volta::fragment<volta::accumulator, 16, 16, 16, float> score_fragment;
  volta::fill_fragment(score_fragment, 0.0f);
  float correction[8] = {};

#pragma unroll
  for (int k_offset = 0; k_offset < kGroupedVerifyHeadDim; k_offset += 16) {
    volta::load_matrix_sync(
        q_fragment, shared_q + m_tile * 16 * kGroupedVerifyQStride + k_offset,
        kGroupedVerifyQStride);
    // K is row-major [N, D].  The same bytes represent K^T as a col-major
    // [D, N] matrix, which is the B operand needed by Q @ K^T.
    volta::load_matrix_sync(
        k_fragment, shared_k + n_tile * 16 * kGroupedVerifyKVStride + k_offset,
        kGroupedVerifyKVStride);
    if constexpr (COMPENSATE) {
      // E4M3 x FP16 products fit comfortably in FP32, but a D256 Tensor
      // Core accumulation can still lose low bits. Sum short K16 products
      // with compensated FP32 additions; explicit RN operations preserve
      // the correction under the standard fast-math build.
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile_fragment;
      volta::fill_fragment(tile_fragment, 0.0f);
      volta::mma_sync(tile_fragment, q_fragment, k_fragment, tile_fragment);
#pragma unroll
      for (int i = 0; i < score_fragment.num_elements; ++i) {
        const float y = __fsub_rn(tile_fragment.x[i], correction[i]);
        const float sum = __fadd_rn(score_fragment.x[i], y);
        correction[i] = __fsub_rn(__fsub_rn(sum, score_fragment.x[i]), y);
        score_fragment.x[i] = sum;
      }
    } else {
      volta::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
    }
  }
#pragma unroll
  for (int i = 0; i < score_fragment.num_elements; ++i) {
    score_fragment.x[i] *= qk_scale;
  }
  volta::store_matrix_sync(
      shared_scores + m_tile * 16 * kGroupedVerifyScoreStride + n_tile * 16,
      score_fragment, kGroupedVerifyScoreStride, volta::mem_row_major);
}

__device__ __forceinline__ void grouped_verify_scale_output_fragment(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& fragment,
    const float* __restrict__ row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
  fragment.x[0] *= first_scale;
  fragment.x[1] *= first_scale;
  fragment.x[2] *= second_scale;
  fragment.x[3] *= second_scale;
  fragment.x[4] *= first_scale;
  fragment.x[5] *= first_scale;
  fragment.x[6] *= second_scale;
  fragment.x[7] *= second_scale;
}

__device__ __forceinline__ void grouped_verify_add_output_tile(
    volta::fragment<volta::accumulator, 16, 16, 16, float>& output,
    const volta::fragment<volta::accumulator, 16, 16, 16, float>& tile,
    const float* row_scale, const int tile_row_start) {
  const int lane = threadIdx.x & 31;
  const int row = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const float first_scale = row_scale[tile_row_start + row];
  const float second_scale = row_scale[tile_row_start + row + 2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float scale = (i & 2) ? second_scale : first_scale;
    output.x[i] = fmaf(output.x[i], scale, tile.x[i]);
  }
}

template <bool SPARSE_PAGE4, bool ROW_SEQLENS = false>
__device__ __forceinline__ bool grouped_verify_key_visible(
    const uint32_t* __restrict__ sparse_tile_masks, const int token_idx,
    const int query_len, const int head_idx, const int kv_idx,
    const int valid_k_rows, const int lane_or_col, const int prefix_kv_len,
    const int* row_lengths = nullptr) {
  const bool row_valid = token_idx < query_len &&
                         head_idx < kGroupedVerifyHeads &&
                         lane_or_col < valid_k_rows;
  if (!row_valid) {
    return false;
  }
  if constexpr (SPARSE_PAGE4) {
    const uint32_t token_mask =
        sparse_tile_masks[(kv_idx & (kGroupedVerifyBlockN - 1)) >> 2];
    const int mask_bit = token_idx * 4 + (kv_idx & 3);
    return (token_mask & (1u << mask_bit)) != 0;
  }
  if constexpr (ROW_SEQLENS) {
    return kv_idx < row_lengths[token_idx];
  }
  return kv_idx <= prefix_kv_len + token_idx;
}

// Grouped sparse verify kernel (ported from flash_decode_paged.cu:2037-2671)
template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE = 0,
          bool SINGLE_QUERY = false, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool STAGE_PARTITION_PAGE_IDS = false,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
          bool SPARSE_PAGE4 = false, typename PARTIAL_T = __half,
          bool ROW_SEQLENS = false, bool COMPENSATE_P = false>
__global__
__launch_bounds__(kGroupedVerifyThreads, 1) void flash_attention_grouped_verify_e5m2_partial_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, PARTIAL_T* __restrict__ partial_out,
    float* __restrict__ partial_lse, const int query_len,
    const int max_num_blocks, const int page_block_size,
    const int pages_per_block,
    const int64_t k_block_stride, const int64_t k_token_stride,
    const int64_t k_head_stride, const int64_t v_block_stride,
    const int64_t v_token_stride, const int64_t v_head_stride,
    const float qk_scale, const float v_scale,
    const uint32_t* __restrict__ sparse_token_masks = nullptr,
    const int num_groups = 1, const int* row_lengths = nullptr,
    const int abl = 0) {
  static_assert(!ROW_SEQLENS || !SPARSE_PAGE4,
                "explicit row lengths apply to dense causal groups only");
  static_assert(
      !COMPENSATE_P || (!TWO_PASS && ROW_SEQLENS &&
                        KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                        std::is_same_v<PARTIAL_T, float>),
      "probability compensation is isolated to E4M3 FP32 groups");
  using Traits = GroupedVerifyTraits<MAX_QUERY_TOKENS>;
  const int head_group = blockIdx.x;
  const int split_id = blockIdx.y;
  const int group_idx = blockIdx.z;
  if (head_group >= Traits::kHeadGroups || split_id >= Traits::kSplits ||
      group_idx >= num_groups || query_len <= 0 ||
      query_len > MAX_QUERY_TOKENS) {
    return;
  }
  if constexpr (ROW_SEQLENS) {
    row_lengths += static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS;
  }

  if constexpr (!SPARSE_PAGE4) {
    partial_out += static_cast<int64_t>(group_idx) * Traits::kSplits *
                   MAX_QUERY_TOKENS * kGroupedVerifyHeads *
                   kGroupedVerifyHeadDim;
    partial_lse += static_cast<int64_t>(group_idx) * Traits::kSplits *
                   MAX_QUERY_TOKENS * kGroupedVerifyHeads *
                   (ROW_SEQLENS ? 2 : 1);
  }

  int total_kv = seq_lens[group_idx];
  if constexpr (ROW_SEQLENS) {
    total_kv = 0;
    for (int i = 0; i < query_len; ++i)
      total_kv = max(total_kv, row_lengths[i]);
  }
  if (total_kv <= 0) {
    if constexpr (SPARSE_PAGE4) {
      constexpr int kGroupOutputElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads * kGroupedVerifyHeadDim;
      const int64_t group_output_offset =
          static_cast<int64_t>(group_idx) * kGroupOutputElements;
      for (int idx = threadIdx.x; idx < kGroupOutputElements;
           idx += kGroupedVerifyThreads) {
        partial_out[group_output_offset + idx] = __float2half_rn(0.0f);
      }
      constexpr int kGroupLseElements =
          kGroupedVerifyQ8MaxQ * kGroupedVerifyHeads;
      if (threadIdx.x < kGroupLseElements) {
        partial_lse[static_cast<int64_t>(group_idx) * kGroupLseElements +
                    threadIdx.x] = kXQANegInf;
      }
    }
    return;
  }
  const int active_splits =
      SPARSE_PAGE4
          ? 1
          : grouped_verify_active_splits<MAX_QUERY_TOKENS, SINGLE_QUERY>(
                total_kv);
  if (split_id >= active_splits) {
    return;
  }
  const int total_tiles =
      (total_kv + kGroupedVerifyBlockN - 1) / kGroupedVerifyBlockN;
  const int base_tiles = total_tiles / active_splits;
  const int extra_tiles = total_tiles - base_tiles * active_splits;
  const int split_tile_start =
      split_id * base_tiles + min(split_id, extra_tiles);
  const int split_tiles = base_tiles + (split_id < extra_tiles ? 1 : 0);
  const int split_start = split_tile_start * kGroupedVerifyBlockN;
  const int split_end =
      min(total_kv, split_start + split_tiles * kGroupedVerifyBlockN);
  const int prefix_kv_len = max(0, total_kv - query_len);
  const int head_start = head_group * Traits::kHeadsPerCta;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  extern __shared__ char grouped_verify_smem_raw[];
  GroupedVerifySmem& smem =
      *reinterpret_cast<GroupedVerifySmem*>(grouped_verify_smem_raw);
  __half* shared_q = smem.storage.compute.q;
  __half* shared_kv = smem.storage.compute.kv;
  float* shared_scores = smem.storage.compute.scores;
  __half* shared_probs = smem.storage.compute.probs;
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_NVFP4) {
    // The first __syncthreads() below publishes the LUT to all threads.
    nvfp4_lut_build(smem.nvfp4_lut, tid, kGroupedVerifyThreads);
  }
  // The compensated entry reserves one padded probability panel after the
  // legacy layout. Match P's shared-memory stride without aliasing live QK
  // rows. Legacy entries retain their existing shared-memory footprint.
  __half* shared_prob_residual = reinterpret_cast<__half*>(
      grouped_verify_smem_raw + sizeof(GroupedVerifySmem));
  constexpr int kResidualStride = kGroupedVerifyProbStride;
  const int* page_ids =
      block_table + static_cast<int64_t>(group_idx) * max_num_blocks;
  int split_page_offset = 0;
  bool use_staged_page_ids = false;
  if constexpr (STAGE_PARTITION_PAGE_IDS) {
    static_assert(PAGE_BLOCK_SIZE > 0,
                  "partition page staging requires a specialized page size");
    const int split_start_page = split_start / PAGE_BLOCK_SIZE;
    split_page_offset = split_start - split_start_page * PAGE_BLOCK_SIZE;
    const int split_page_count =
        (split_page_offset + split_end - split_start + PAGE_BLOCK_SIZE - 1) /
        PAGE_BLOCK_SIZE;
    use_staged_page_ids = split_page_count <= kGroupedVerifyPageIdsCapacity;
    if (use_staged_page_ids) {
      for (int idx = tid; idx < split_page_count;
           idx += kGroupedVerifyThreads) {
        smem.page_ids[idx] = __ldg(&page_ids[split_start_page + idx]);
      }
    }
    if (use_staged_page_ids) {
      page_ids = smem.page_ids;
    }
  }

  constexpr int kVecsPerRow = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedQVecsPerRow = kGroupedVerifyQStride / 8;
  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* shared_q_vec = reinterpret_cast<uint4*>(shared_q);
  for (int idx = tid; idx < kGroupedVerifyRows * kVecsPerRow;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kVecsPerRow;
    const int vec_col = idx % kVecsPerRow;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const int64_t query_row =
          static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = __ldg(
          q_vec + (query_row * kGroupedVerifyHeads + head_idx) * kVecsPerRow +
          vec_col);
    } else {
      shared_q_vec[row * kSharedQVecsPerRow + vec_col] = make_uint4(0, 0, 0, 0);
    }
  }
  if (tid < kGroupedVerifyRows) {
    smem.row_max[tid] = kXQANegInf;
    smem.row_sum[tid] = 0.0f;
    smem.row_scale[tid] = 1.0f;
  }
  // This barrier publishes both Q and the optional split-local page IDs.
  __syncthreads();

  constexpr int kPanelStrideVec = kGroupedVerifyHeadDim / 8;
  constexpr int kSharedStrideVec = kGroupedVerifyKVStride / 8;

  volta::fragment<volta::accumulator, 16, 16, 16, float>
      output_fragments[kGroupedVerifyOutputTilesPerWarp];
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    volta::fill_fragment(output_fragments[fragment_idx], 0.0f);
  }

  // The conservative baseline computes exact per-split FP32 max/sum in a
  // separate pass. The candidate below fuses this state update with P x V.
  if constexpr (TWO_PASS) {
    for (int tile_start = split_start; tile_start < split_end;
         tile_start += kGroupedVerifyBlockN) {
      const int valid_k_rows =
          min(kGroupedVerifyBlockN, split_end - tile_start);
      if constexpr (SPARSE_PAGE4) {
        const int valid_sparse_pages = (valid_k_rows + 3) / 4;
        if (tid < kGroupedVerifyBlockN / 4) {
          smem.sparse_token_masks[tid] =
              tid < valid_sparse_pages
                  ? __ldg(sparse_token_masks +
                          static_cast<int64_t>(group_idx) * max_num_blocks +
                          (tile_start >> 2) + tid)
                  : 0;
        }
      }
      const int tile_page_offset =
          use_staged_page_ids ? split_page_offset + tile_start - split_start
                              : tile_start;
      load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                           kGroupedVerifyThreads, KV_DTYPE,
                           KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                               (KV_DTYPE ==
                                    flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                                !ROW_SEQLENS)>(
          shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
          kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
          k_block_stride, k_token_stride, k_head_stride, 0, pages_per_block,
          threadIdx.x,
          nullptr, smem.nvfp4_lut, abl);
      for (int idx = tid + valid_k_rows * kSharedStrideVec;
           idx < kGroupedVerifyBlockN * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();
      // W12: defensively zero K rows no query attends (see the V panel below).
      // K NaN is already masked out of the scores by
      // grouped_verify_key_visible, but zeroing keeps the loaded panel free of
      // the null block's E4M3 NaN. Masks are visible after the __syncthreads()
      // above; the one added inside publishes the zeros before
      // grouped_verify_qk consumes shared_kv.
      if constexpr (SPARSE_PAGE4) {
        for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
             idx += kGroupedVerifyThreads) {
          const int row = idx / kSharedStrideVec;
          const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
          if ((token_mask & (0x11111111u << (row & 3))) == 0) {
            reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
          }
        }
        __syncthreads();
      }

      int active_m_tiles = 0x7;
      if constexpr (SPARSE_PAGE4) {
        uint32_t active_query_nibbles = 0;
#pragma unroll
        for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
          active_query_nibbles |= smem.sparse_token_masks[page];
        }
        active_m_tiles = 0;
#pragma unroll
        for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
          if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
            const int first_row = token * kGroupedVerifyHeads;
            const int last_row = first_row + kGroupedVerifyHeads - 1;
            active_m_tiles |= 1 << (first_row / 16);
            active_m_tiles |= 1 << (last_row / 16);
          }
        }
      }
      grouped_verify_qk(shared_q, shared_kv, shared_scores, qk_scale,
                        active_m_tiles);
      __syncthreads();

#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float probability =
            visible ? __expf(fmaxf(score - tile_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        if (lane_id == 0 && tile_sum > 0.0f) {
          const float old_max = smem.row_max[row];
          const float old_sum = smem.row_sum[row];
          const float new_max = fmaxf(old_max, tile_max);
          smem.row_sum[row] =
              old_sum * __expf(fmaxf(old_max - new_max, -80.0f)) +
              tile_sum * __expf(fmaxf(tile_max - new_max, -80.0f));
          smem.row_max[row] = new_max;
        }
      }
      __syncthreads();
    }
  }

  // Recompute QK for the conservative path, or consume it once while updating
  // the online state for the fused path. Both form half P and use identical
  // Volta WMMA P @ V arithmetic.
  for (int tile_start = split_start; tile_start < split_end;
       tile_start += kGroupedVerifyBlockN) {
    const int valid_k_rows = min(kGroupedVerifyBlockN, split_end - tile_start);
    if constexpr (SPARSE_PAGE4) {
      const int valid_sparse_pages = (valid_k_rows + 3) / 4;
      if (tid < kGroupedVerifyBlockN / 4) {
        smem.sparse_token_masks[tid] =
            tid < valid_sparse_pages
                ? __ldg(sparse_token_masks +
                        static_cast<int64_t>(group_idx) * max_num_blocks +
                        (tile_start >> 2) + tid)
                : 0;
      }
    }
    const int tile_page_offset =
        use_staged_page_ids ? split_page_offset + tile_start - split_start
                            : tile_start;
    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE,
                         KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                             (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                              !ROW_SEQLENS)>(
        shared_kv, k_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        k_block_stride, k_token_stride, k_head_stride, 0, pages_per_block,
        threadIdx.x,
        nullptr, smem.nvfp4_lut, abl);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    // W12: defensively zero K rows no query attends (see the V panel below).
    // K NaN is already masked out of the scores by grouped_verify_key_visible,
    // but zeroing keeps the loaded panel free of the null block's E4M3 NaN.
    // Masks are visible after the __syncthreads() above; the one added inside
    // publishes the zeros before grouped_verify_qk consumes shared_kv.
    if constexpr (SPARSE_PAGE4) {
      for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kSharedStrideVec;
        const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
        if ((token_mask & (0x11111111u << (row & 3))) == 0) {
          reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
        }
      }
      __syncthreads();
    }

    int active_m_tiles = 0x7;
    if constexpr (COMPENSATE_P) {
      // Native MTP4 supplies five rows: only 30 of the 48 packed rows are
      // live. Avoid QK/PV for wholly padded M tiles to pay for the residual
      // product without changing arithmetic in any live row.
      active_m_tiles =
          (1 << ((query_len * Traits::kHeadsPerCta + 15) / 16)) - 1;
    }
    if constexpr (SPARSE_PAGE4) {
      uint32_t active_query_nibbles = 0;
#pragma unroll
      for (int page = 0; page < kGroupedVerifyBlockN / 4; ++page) {
        active_query_nibbles |= smem.sparse_token_masks[page];
      }
      active_m_tiles = 0;
#pragma unroll
      for (int token = 0; token < kGroupedVerifyQ8MaxQ; ++token) {
        if ((active_query_nibbles & (0xFu << (token * 4))) != 0) {
          const int first_row = token * kGroupedVerifyHeads;
          const int last_row = first_row + kGroupedVerifyHeads - 1;
          active_m_tiles |= 1 << (first_row / 16);
          active_m_tiles |= 1 << (last_row / 16);
        }
      }
    }
    grouped_verify_qk<COMPENSATE_P>(shared_q, shared_kv, shared_scores,
                                    qk_scale, active_m_tiles);
    __syncthreads();

    if constexpr (TWO_PASS) {
      for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyBlockN;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kGroupedVerifyBlockN;
        const int col = idx % kGroupedVerifyBlockN;
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + col;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, col, prefix_kv_len, row_lengths) &&
            smem.row_sum[row] > 0.0f;
        const float probability =
            visible ? __expf(fmaxf(
                          shared_scores[row * kGroupedVerifyScoreStride + col] -
                              smem.row_max[row],
                          -80.0f))
                    : 0.0f;
        shared_probs[row * kGroupedVerifyProbStride + col] =
            __float2half_rn(probability);
      }
      __syncthreads();
    } else {
#pragma unroll
      for (int row = warp_id; row < kGroupedVerifyRows;
           row += kGroupedVerifyWarps) {
        if constexpr (COMPENSATE_P) {
          if ((active_m_tiles & (1 << (row / 16))) == 0) {
            continue;
          }
        }
        const int token_idx = row / Traits::kHeadsPerCta;
        const int local_head = row % Traits::kHeadsPerCta;
        const int head_idx = head_start + local_head;
        const int kv_idx = tile_start + lane_id;
        const bool visible =
            grouped_verify_key_visible<SPARSE_PAGE4, ROW_SEQLENS>(
                smem.sparse_token_masks, token_idx, query_len, head_idx, kv_idx,
                valid_k_rows, lane_id, prefix_kv_len, row_lengths);
        const float score =
            visible ? shared_scores[row * kGroupedVerifyScoreStride + lane_id]
                    : kXQANegInf;
        const float tile_max_lane = warp_reduce_max(score);
        const float tile_max = __shfl_sync(0xffffffffu, tile_max_lane, 0);
        const float old_max = smem.row_max[row];
        const float new_max = fmaxf(old_max, tile_max);
        const float probability =
            visible ? __expf(fmaxf(score - new_max, -80.0f)) : 0.0f;
        const float tile_sum_lane = warp_reduce_sum(probability);
        const float tile_sum = __shfl_sync(0xffffffffu, tile_sum_lane, 0);
        const float exp_diff =
            tile_sum > 0.0f ? __expf(fmaxf(old_max - new_max, -80.0f)) : 1.0f;
        shared_probs[row * kGroupedVerifyProbStride + lane_id] =
            __float2half_rn(probability);
        if constexpr (COMPENSATE_P) {
          const float rounded = __half2float(__float2half_rn(probability));
          shared_prob_residual[row * kResidualStride + lane_id] =
              __float2half_rn((probability - rounded) * 2048.0f);
        }
        // Finish every lane's shared-state reads before lane 0 overwrites the
        // online maximum. Shuffle synchronization does not order memory.
        __syncwarp();
        if (lane_id == 0) {
          if (tile_sum > 0.0f) {
            smem.row_sum[row] = smem.row_sum[row] * exp_diff + tile_sum;
            smem.row_max[row] = new_max;
          }
          smem.row_scale[row] = exp_diff;
        }
      }
      __syncthreads();
      if constexpr (!COMPENSATE_P) {
#pragma unroll
        for (int fragment_idx = 0;
             fragment_idx < kGroupedVerifyOutputTilesPerWarp; ++fragment_idx) {
          const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
          const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
          grouped_verify_scale_output_fragment(output_fragments[fragment_idx],
                                               smem.row_scale, m_tile * 16);
        }
      }
    }

    load_xqa_tc_kv_panel<PAGE_BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,
                         kGroupedVerifyThreads, KV_DTYPE,
                         KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 ||
                             (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E4M3 &&
                              !ROW_SEQLENS)>(
        shared_kv, v_cache, page_ids, valid_k_rows, kPanelStrideVec,
        kSharedStrideVec, tile_page_offset, 0, page_block_size, 0,
        v_block_stride, v_token_stride, v_head_stride, 0, pages_per_block,
        threadIdx.x,
        nullptr, smem.nvfp4_lut, abl);
    for (int idx = tid + valid_k_rows * kSharedStrideVec;
         idx < kGroupedVerifyBlockN * kSharedStrideVec;
         idx += kGroupedVerifyThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    // W12: the grouped planner pads each category to a multiple of 8 with
    // (physical microblock 0 = null block, mask 0) and counts the padding in
    // seq_len; the forward also carries unselected tokens inside partial
    // microblocks. Such rows get P=0, but 0 * NaN survives the P@V MMA when
    // page 0 holds fp16 GDN state that decodes to E4M3 NaN (W8). Zero every
    // row no query attends (mask bit = query*4 + token_in_microblock, so the
    // row is unattended iff (mask & (0x11111111u << (row & 3))) == 0) so the
    // reduction sees 0 * 0 = 0. The leading __syncthreads() makes the V panel
    // load finish before we overwrite its page-0 rows (else the load races past
    // the zero and resurrects the NaN); the trailing one publishes the zeros
    // before P@V. Masks were synced earlier in this iteration.
    if constexpr (SPARSE_PAGE4) {
      __syncthreads();
      for (int idx = tid; idx < valid_k_rows * kSharedStrideVec;
           idx += kGroupedVerifyThreads) {
        const int row = idx / kSharedStrideVec;
        const uint32_t token_mask = smem.sparse_token_masks[row >> 2];
        if ((token_mask & (0x11111111u << (row & 3))) == 0) {
          reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
         ++fragment_idx) {
      const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
      const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
      const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
      if constexpr (SPARSE_PAGE4 || COMPENSATE_P) {
        if ((active_m_tiles & (1 << m_tile)) == 0) {
          continue;
        }
      }
      volta::fragment<volta::matrix_a, 16, 16, 16, half, volta::row_major>
          probability_fragment;
      volta::fragment<volta::matrix_b, 16, 16, 16, half, volta::row_major>
          value_fragment;
      volta::fragment<volta::accumulator, 16, 16, 16, float> tile_fragment;
      if constexpr (COMPENSATE_P) {
        // Keep Tensor Core accumulation local to N32. Carrying a large C
        // through every KV tile loses small PV corrections at long contexts.
        // The online state remains FP32 and is updated once per tile below.
        volta::fill_fragment(tile_fragment, 0.0f);
      }
      auto& pv_fragment =
          COMPENSATE_P ? tile_fragment : output_fragments[fragment_idx];
#pragma unroll
      for (int k_offset = 0; k_offset < kGroupedVerifyBlockN; k_offset += 16) {
        volta::load_matrix_sync(
            probability_fragment,
            shared_probs + m_tile * 16 * kGroupedVerifyProbStride + k_offset,
            kGroupedVerifyProbStride);
        volta::load_matrix_sync(
            value_fragment,
            shared_kv + k_offset * kGroupedVerifyKVStride + d_tile * 16,
            kGroupedVerifyKVStride);
        volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                        pv_fragment);
        if constexpr (COMPENSATE_P) {
          // Scale the residual up before its FP16 conversion so small
          // corrections do not underflow. Every finite E4M3 value divided
          // by 2048 is still exactly representable in FP16, so applying
          // the inverse power of two to V introduces no operand rounding.
          volta::load_matrix_sync(
              probability_fragment,
              shared_prob_residual + m_tile * 16 * kResidualStride + k_offset,
              kResidualStride);
#pragma unroll
          for (int i = 0; i < value_fragment.num_elements / 2; ++i) {
            union {
              uint32_t bits;
              __half2 pair;
            } packed_value;
            packed_value.bits = value_fragment.x[i];
            packed_value.pair =
                __hmul2(packed_value.pair, __float2half2_rn(1.0f / 2048.0f));
            value_fragment.x[i] = packed_value.bits;
          }
          volta::mma_sync(pv_fragment, probability_fragment, value_fragment,
                          pv_fragment);
        }
      }
      if constexpr (COMPENSATE_P) {
        grouped_verify_add_output_tile(output_fragments[fragment_idx],
                                       tile_fragment, smem.row_scale,
                                       m_tile * 16);
      }
    }
    __syncthreads();
  }

  // The compute buffers are dead. Reuse their storage for the dense FP32
  // partial output, then normalize and write only real query/head rows.
  __syncthreads();
  float* shared_output = smem.storage.output;
#pragma unroll
  for (int fragment_idx = 0; fragment_idx < kGroupedVerifyOutputTilesPerWarp;
       ++fragment_idx) {
    const int output_tile = warp_id + fragment_idx * kGroupedVerifyWarps;
    const int m_tile = output_tile / (kGroupedVerifyHeadDim / 16);
    const int d_tile = output_tile % (kGroupedVerifyHeadDim / 16);
    volta::store_matrix_sync(
        shared_output + m_tile * 16 * kGroupedVerifyHeadDim + d_tile * 16,
        output_fragments[fragment_idx], kGroupedVerifyHeadDim,
        volta::mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < kGroupedVerifyRows * kGroupedVerifyHeadDim;
       idx += kGroupedVerifyThreads) {
    const int row = idx / kGroupedVerifyHeadDim;
    const int d = idx % kGroupedVerifyHeadDim;
    const int token_idx = row / Traits::kHeadsPerCta;
    const int local_head = row % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[row];
      // Explicit-row FP32 groups retain the numerator until the final merge.
      // Normalizing each partition and encoding its weight as m+log(sum)
      // introduces avoidable rounding, including at FP16 output midpoints.
      const float scale =
          sum > 0.0f ? (ROW_SEQLENS ? v_scale : v_scale / sum) : 0.0f;
      int64_t output_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        output_idx = (global_token_idx * kGroupedVerifyHeads + head_idx) *
                         kGroupedVerifyHeadDim +
                     d;
      } else {
        output_idx =
            (((static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                  kGroupedVerifyHeads +
              head_idx) *
                 kGroupedVerifyHeadDim +
             d);
      }
      if constexpr (std::is_same_v<PARTIAL_T, float>) {
        partial_out[output_idx] = shared_output[idx] * scale;
      } else {
        partial_out[output_idx] = __float2half_rn(shared_output[idx] * scale);
      }
    }
  }
  if (tid < kGroupedVerifyRows) {
    const int token_idx = tid / Traits::kHeadsPerCta;
    const int local_head = tid % Traits::kHeadsPerCta;
    const int head_idx = head_start + local_head;
    if (token_idx < query_len && head_idx < kGroupedVerifyHeads) {
      const float sum = smem.row_sum[tid];
      int64_t lse_idx;
      if constexpr (SPARSE_PAGE4) {
        const int64_t global_token_idx =
            static_cast<int64_t>(group_idx) * MAX_QUERY_TOKENS + token_idx;
        lse_idx = global_token_idx * kGroupedVerifyHeads + head_idx;
      } else {
        lse_idx =
            (static_cast<int64_t>(split_id) * MAX_QUERY_TOKENS + token_idx) *
                kGroupedVerifyHeads +
            head_idx;
      }
      if constexpr (ROW_SEQLENS) {
        partial_lse[2 * lse_idx] = sum > 0.0f ? smem.row_max[tid] : kXQANegInf;
        partial_lse[2 * lse_idx + 1] = sum;
      } else {
        partial_lse[lse_idx] =
            sum > 0.0f ? smem.row_max[tid] + logf(sum) : kXQANegInf;
      }
    }
  }
}

// Host binding: transcribed from flash_decode_paged.cu:4834-4946 with the
// stable-ABI API conversion (device code above is unchanged).
torch::stable::Tensor flash_attention_grouped_sparse_page4(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& token_masks,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& lse,
    // The stable-ABI schema "float" binds to C++ double (in-tree
    // convention, see sm70_grouped_verify_nvfp4); the fp32 kernel path
    // casts at the launch site.
    const double softmax_scale, const std::string& kv_cache_dtype,
    const double k_scale, const double v_scale) {
  constexpr int kQueriesPerGroup = kGroupedVerifyQ8MaxQ;
  STD_TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                  "grouped sparse page4 q/K/V must be CUDA tensors");
  STD_TORCH_CHECK(block_table.is_cuda() && token_masks.is_cuda() &&
                      seq_lens.is_cuda() && lse.is_cuda(),
                  "grouped sparse page4 metadata must be CUDA tensors");
  // ABL channel: the offline experiment passes e.g. "nvfp4@3". The suffix is
  // stripped here (schema unchanged, so the production binding is untouched);
  // abl == 0 is byte-identical to production.
  std::string kv_dtype = kv_cache_dtype;
  int abl = 0;  // offline ABL knob; 0 = byte-identical to production
  if (const auto at = kv_dtype.find('@'); at != std::string::npos) {
    abl = std::atoi(kv_dtype.c_str() + at + 1);
    kv_dtype = kv_dtype.substr(0, at);
  }
  const bool e4m3_kv = kv_dtype == "fp8" || kv_dtype == "fp8_e4m3";
  const bool e5m2_kv = kv_dtype == "fp8_e5m2";
  const bool nvfp4_kv = kv_dtype == "nvfp4";
  // All three pack the KV into uint8 storage that the kernel dequantizes in
  // flight; the layer scale folds host-side into qk/v exactly as before.
  const bool packed_kv = e4m3_kv || e5m2_kv || nvfp4_kv;
  STD_TORCH_CHECK(q.scalar_type() == torch::headeronly::ScalarType::Half,
                  "grouped sparse page4 requires fp16 queries");
  STD_TORCH_CHECK(
      packed_kv ? (k_cache.scalar_type() == torch::headeronly::ScalarType::Byte &&
                   v_cache.scalar_type() == torch::headeronly::ScalarType::Byte)
                : (k_cache.scalar_type() ==
                       torch::headeronly::ScalarType::Half &&
                   v_cache.scalar_type() ==
                       torch::headeronly::ScalarType::Half),
      "grouped sparse page4 KV storage does not match kv_cache_dtype");
  STD_TORCH_CHECK(
      packed_kv || kv_dtype == "auto" || kv_dtype == "float16",
      "grouped sparse page4 supports fp16, fp8_e4m3, fp8_e5m2 and nvfp4 KV");
  STD_TORCH_CHECK(
      !packed_kv || (std::isfinite(k_scale) && std::isfinite(v_scale) &&
                     k_scale > 0.0f && v_scale > 0.0f),
      "grouped sparse page4 packed KV scales must be finite and positive");
  STD_TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int &&
          seq_lens.scalar_type() == torch::headeronly::ScalarType::Int &&
          token_masks.scalar_type() ==
              torch::headeronly::ScalarType::UInt32,
      "grouped sparse page4 tables must be int32/uint32");
  STD_TORCH_CHECK(q.dim() == 3 && q.sizes()[0] > 0 &&
                      q.sizes()[0] % kQueriesPerGroup == 0 &&
                      q.sizes()[1] == kGroupedVerifyHeads &&
                      q.sizes()[2] == kGroupedVerifyHeadDim,
                  "grouped sparse page4 q must have shape [8*N, 6, 256]");
  STD_TORCH_CHECK(
      k_cache.dim() == 4 && v_cache.dim() == 4 &&
          k_cache.sizes() == v_cache.sizes() && k_cache.sizes()[2] == 1 &&
          (nvfp4_kv
               ? (k_cache.sizes()[1] % 4 == 0 &&
                  k_cache.sizes()[3] == kNvfp4CellBytes &&
                  k_cache.strides()[0] >=
                      k_cache.sizes()[1] * kNvfp4TokenCellBytes &&
                  k_cache.strides()[1] == kNvfp4TokenCellBytes &&
                  k_cache.strides()[2] == kNvfp4CellBytes)
               : (k_cache.sizes()[1] == 4 &&
                  k_cache.sizes()[3] == kGroupedVerifyHeadDim)),
      "grouped sparse page4 KV must have shape [blocks, 4, 1, 256] (nvfp4: "
      "the [pool_blocks, N, 1, 144] cell pool view)");
  const int64_t num_groups = q.sizes()[0] / kQueriesPerGroup;
  STD_TORCH_CHECK(
      block_table.dim() == 2 && block_table.sizes()[0] == num_groups &&
          token_masks.sizes() == block_table.sizes(),
      "grouped sparse page4 block IDs/masks must be [groups, pages]");
  // Restructured from `seq_lens.sizes() == at::IntArrayRef({num_groups})`:
  // the stable ABI has no IntArrayRef, so check dim + element directly.
  STD_TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.sizes()[0] == num_groups,
                  "grouped sparse page4 seq_lens must have shape [groups]");
  STD_TORCH_CHECK(q.is_contiguous() && block_table.is_contiguous() &&
                      token_masks.is_contiguous() && seq_lens.is_contiguous(),
                  "grouped sparse page4 q/metadata must be contiguous");
  STD_TORCH_CHECK(k_cache.strides()[k_cache.dim() - 1] == 1 &&
                      v_cache.strides()[v_cache.dim() - 1] == 1,
                  "grouped sparse page4 KV head dimension must be contiguous");
  // Restructured from
  // `lse.sizes() == at::IntArrayRef({q.sizes()[0], kGroupedVerifyHeads})`.
  STD_TORCH_CHECK(
      lse.dim() == 2 && lse.sizes()[0] == q.sizes()[0] &&
          lse.sizes()[1] == kGroupedVerifyHeads &&
          lse.scalar_type() == torch::headeronly::ScalarType::Float &&
          lse.is_contiguous(),
      "grouped sparse page4 lse must be contiguous [rows, 6] fp32");

  // Restructured: the stable ABI has no torch::empty_like / out-device
  // comparison, so the output is caller-provided (in-tree convention) and
  // validated here (device compared via get_device_index()).
  STD_TORCH_CHECK(
      out.is_cuda() && out.get_device_index() == q.get_device_index() &&
          out.scalar_type() == torch::headeronly::ScalarType::Half &&
          out.sizes() == q.sizes() && out.is_contiguous(),
      "grouped sparse page4 out must be contiguous fp16 and q-shaped");
  STD_TORCH_CHECK(
      q.get_device_index() == k_cache.get_device_index() &&
          q.get_device_index() == v_cache.get_device_index() &&
          q.get_device_index() == block_table.get_device_index() &&
          q.get_device_index() == token_masks.get_device_index() &&
          q.get_device_index() == seq_lens.get_device_index() &&
          q.get_device_index() == lse.get_device_index() &&
          q.get_device_index() == out.get_device_index(),
      "all grouped sparse page4 tensors must be on the same device");

  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  const auto* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 7 && properties->minor == 0,
                  "grouped sparse page4 supports SM70 only");
  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());
  const dim3 grid(1, 1, static_cast<unsigned>(num_groups));
  const size_t shared_mem = sizeof(GroupedVerifySmem);
  // The NVFP4 reader decodes 4-token microblocks (pool_block *
  // pages_per_block + page_in_block); pages_per_block is the pool block's
  // token count / 4 (shape-driven: allocator pages are not dense multiples
  // of the cell). Other dtypes ignore the value.
  const int nvfp4_pages_per_block =
      nvfp4_kv ? static_cast<int>(k_cache.sizes()[1] / 4) : 0;
#define LAUNCH_GROUPED_SPARSE_PAGE4(KV_DTYPE_CODE)                            \
  do {                                                                        \
    auto kernel =                                                            \
        (void*)flash_attention_grouped_verify_e5m2_partial_kernel<            \
            kQueriesPerGroup, false, 4, false, false, false, KV_DTYPE_CODE,   \
            true>;                                                            \
    const cudaError_t smem_status = cudaFuncSetAttribute(                     \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);     \
    STD_TORCH_CHECK(                                                          \
        smem_status == cudaSuccess,                                           \
        "Failed to set grouped sparse page4 shared memory: ",                  \
        cudaGetErrorString(smem_status));                                     \
    flash_attention_grouped_verify_e5m2_partial_kernel<                        \
        kQueriesPerGroup, false, 4, false, false, false, KV_DTYPE_CODE, true>  \
        <<<grid, kGroupedVerifyThreads, shared_mem, stream>>>(                \
            reinterpret_cast<const __half*>(q.const_data_ptr()),              \
            k_cache.const_data_ptr(), v_cache.const_data_ptr(),               \
            block_table.const_data_ptr<int>(),                                \
            seq_lens.const_data_ptr<int>(),                                   \
            reinterpret_cast<__half*>(out.mutable_data_ptr()),                \
            lse.mutable_data_ptr<float>(),                                    \
            kQueriesPerGroup, static_cast<int>(block_table.sizes()[1]), 4,     \
            nvfp4_pages_per_block,                                             \
            k_cache.strides()[0], k_cache.strides()[1], k_cache.strides()[2],  \
            v_cache.strides()[0], v_cache.strides()[1], v_cache.strides()[2],  \
            static_cast<float>(softmax_scale) *                               \
                (packed_kv ? static_cast<float>(k_scale) : 1.0f),             \
            packed_kv ? static_cast<float>(v_scale) : 1.0f,                   \
            token_masks.const_data_ptr<uint32_t>(),                           \
            static_cast<int>(num_groups), nullptr, abl);                                    \
  } while (0)
  if (e4m3_kv) {
    LAUNCH_GROUPED_SPARSE_PAGE4(flash_v100::KV_CACHE_DTYPE_FP8_E4M3);
  } else if (e5m2_kv) {
    LAUNCH_GROUPED_SPARSE_PAGE4(flash_v100::KV_CACHE_DTYPE_FP8_E5M2);
  } else if (nvfp4_kv) {
    LAUNCH_GROUPED_SPARSE_PAGE4(flash_v100::KV_CACHE_DTYPE_NVFP4);
  } else {
    LAUNCH_GROUPED_SPARSE_PAGE4(flash_v100::KV_CACHE_DTYPE_FP16);
  }
#undef LAUNCH_GROUPED_SPARSE_PAGE4
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "grouped sparse page4 launch failed: ",
                  cudaGetErrorString(err));
  return out;
}

}  // namespace qsa_page4
}  // namespace vllm
