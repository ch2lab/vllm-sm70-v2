// SM70 grouped verify attention (DFlash2/MTP 8-position verify), NVFP4 paged KV.
//
// Port of 1Cat-vLLM's production grouped-verify kernel
// (flash-attention-v100/kernel/flash_decode_paged.cu,
// flash_attention_grouped_verify_paged) for our QUASAR-NVFP4 stack:
//   - same 80-split x 512-thread CTA design (one CTA per V100 SM, KV split
//     into <=80 chunks, each CTA: 48 packed rows = 8 query tokens x 6 GQA
//     heads over one KV head, d=256)
//   - two-pass online softmax (exact per-split max/sum, then P@V)
//   - E4M3 KV load replaced by NVFP4 unpack: 4-bit e2m1 nibbles + per-16-dim
//     e4m3 block scales, using the same e2m1/e4m3 tables as sm70_flashdecode
//     (layer k_scale/v_scale baked into the scale table, sm_scale folded into
//     the Q load)
//   - paged layout [num_pages, 2*H, page_tokens, 144] page-first (same as
//     sm70_flashdecode_hmma), block_table int32
//
// Output merge: per-split normalized fp16 partials + fp32 LSE, merged by the
// combine kernel (exact online-softmax LSE merge).
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "sm70_volta_mma.h"

namespace sm70gv {

using namespace sm70gv::volta;

// ---- shape constants (GQA 6:1, d=256; 48 packed rows = 8 q x 6 heads) ----
constexpr int kMaxQ = 8;
constexpr int kHeads = 6;
constexpr int kHeadDim = 256;
constexpr int kRows = 48;                  // kMaxQ * kHeads
constexpr int kBlockN = 32;                // KV tile
constexpr int kQStride = 264;             // padded 256
constexpr int kKVStride = 264;
constexpr int kScoreStride = 32;
constexpr int kProbStride = 40;
constexpr int kSplits = 80;               // one CTA per SM
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;     // 16
constexpr int kQKWarps = (kRows / 16) * (kBlockN / 16);  // 3 * 2 = 6
constexpr int kOutputTiles = (kRows / 16) * (kHeadDim / 16);  // 3 * 16 = 48
constexpr int kOutputTilesPerWarp = kOutputTiles / kWarps;    // 3
constexpr int kMinTokensPerSplit = 64;
constexpr int kShortContextMaxTokens = 128;
constexpr int kKVBytes = 144;              // 128 code bytes + 16 scale bytes
constexpr int kCodeBytes = 128;
constexpr int kScaleBytes = 16;
constexpr float kNegInf = -1.0e30f;

// NVFP4 e2m1 nibble -> value (mirrors sm70_flashdecode_hmma::e2m1_f).
__device__ __forceinline__ float e2m1_f(unsigned code) {
  const unsigned m = code & 7;
  float mag;
  if (m < 5) {
    mag = m * 0.5f;
  } else if (m < 7) {
    mag = (m == 5) ? 3.0f : 4.0f;
  } else {
    mag = 6.0f;
  }
  return (code & 8) ? -mag : mag;
}

__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  const bool sgn = b & 0x80;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  float v;
  if (e == 0) {
    v = m * 0.001953125f;
  } else {
    v = (1.0f + m * 0.125f) * exp2f((float)e - 7.0f);
  }
  return sgn ? -v : v;
}

struct alignas(256) Smem {
  union {
    struct {
      alignas(16) __half q[kRows * kQStride];
      alignas(16) __half kv[kBlockN * kKVStride];
      alignas(16) float scores[kRows * kScoreStride];
      alignas(16) __half probs[kRows * kProbStride];
    } compute;
    alignas(16) float output[kRows * kHeadDim];
  } storage;
  alignas(16) float row_max[kRows];
  alignas(16) float row_sum[kRows];
  alignas(16) float e2t_tab[16];   // nibble table (e2m1 values)
  alignas(16) float scale_tab[2 * 256];  // e4m3 byte * layer scale (K, V)
};

static_assert(sizeof(Smem) <= 64 * 1024,
              "grouped verify must fit Volta's 64 KiB opt-in budget");

__device__ __forceinline__ int active_splits(const int total_kv) {
  const int splits =
      min(kSplits, max(1, (total_kv + kMinTokensPerSplit - 1) /
                              kMinTokensPerSplit));
  return total_kv <= kShortContextMaxTokens ? 1 : splits;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

// QK: scores[48 rows, 32 KV] = q[48, 256] @ k[32, 256]^T, 6 warps on
// 16x16 score tiles (3 row tiles x 2 KV tiles).
__device__ __forceinline__ void grouped_verify_qk(
    const __half* __restrict__ shared_q, const __half* __restrict__ shared_k,
    float* __restrict__ shared_scores) {
  const int warp_id = threadIdx.x / 32;
  if (warp_id >= kQKWarps) {
    return;
  }
  const int m_tile = warp_id / (kBlockN / 16);
  const int n_tile = warp_id % (kBlockN / 16);
  fragment<matrix_a, 16, 16, 16, half, row_major> q_fragment;
  fragment<matrix_b, 16, 16, 16, half, col_major> k_fragment;
  fragment<accumulator, 16, 16, 16, float> score_fragment;
  fill_fragment(score_fragment, 0.0f);
#pragma unroll
  for (int k_offset = 0; k_offset < kHeadDim; k_offset += 16) {
    load_matrix_sync(
        q_fragment, shared_q + m_tile * 16 * kQStride + k_offset, kQStride);
    // K panel is [kv_token, dim] row-major; the same bytes read transposed
    // are the col-major B operand of Q @ K^T.
    load_matrix_sync(
        k_fragment, shared_k + n_tile * 16 * kKVStride + k_offset, kKVStride);
    mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
  }
  store_matrix_sync(
      shared_scores + m_tile * 16 * kScoreStride + n_tile * 16,
      score_fragment, kScoreStride, mem_row_major);
}

// Load one 32-token KV tile (K or V side) from NVFP4 paged cache into the
// fp16 panel shared_kv[kv_token, dim] (row stride kKVStride). One 16-dim
// scale group (8 code bytes + 1 scale byte) per thread: 32 * 16 = 512 items
// == kThreads, no loop.
// NVFP4 paged cache byte layout per (page, side): a contiguous code block
// [H][n_page][128 code bytes] followed by a scale block [H][n_page][16
// scale bytes] (same packing sm70_flashdecode_hmma reads: head h's token t
// codes at h*n_page*128 + t*128, scales at H*n_page*128 + h*n_page*16 + t*16).
// One thread per (kv_token, 16-dim scale group): 32 * 16 = 512 = kThreads.
__device__ __forceinline__ void load_kv_panel(
    const uint8_t* __restrict__ kv_cache, const int* __restrict__ block_table,
    __half* __restrict__ shared_kv, const float* __restrict__ e2t_tab,
    const float* __restrict__ scale_tab, const int r, const int h,
    const int H, const int side, const int tile_start, const int valid_rows,
    const int n_page, const int bt_cols, const int64_t page_stride) {
  const int tid = threadIdx.x;
  const int row = tid / (kHeadDim / 16);       // 0..31
  const int group = tid % (kHeadDim / 16);     // 0..15 (16-dim groups)
  if (row >= valid_rows) {
    return;
  }
  const int tok = tile_start + row;
  const int blk = tok / n_page;
  const int tok_in = tok - blk * n_page;
  const int page = __ldg(block_table + r * bt_cols + blk);
  const int64_t side_base =
      (int64_t)page * page_stride + side * H * (int64_t)n_page * kKVBytes;
  const uint8_t* codes =
      kv_cache + side_base + (int64_t)h * (int64_t)n_page * kCodeBytes +
      (int64_t)tok_in * kCodeBytes + group * 8;
  const uint8_t* scales =
      kv_cache + side_base + (int64_t)H * (int64_t)n_page * kCodeBytes +
      (int64_t)h * (int64_t)n_page * kScaleBytes +
      (int64_t)tok_in * kScaleBytes + group;
  unsigned short words[4];
  {
    const uint2 c = *reinterpret_cast<const uint2*>(codes);
    words[0] = (unsigned short)(c.x & 0xFFFF);
    words[1] = (unsigned short)(c.x >> 16);
    words[2] = (unsigned short)(c.y & 0xFFFF);
    words[3] = (unsigned short)(c.y >> 16);
  }
  const float sf = scale_tab[side * 256 + __ldg(scales)];
  // word w holds bytes 2w, 2w+1 = dims 4w..4w+3 (lo nibble first).
  __half* out = shared_kv + row * kKVStride + group * 16;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const unsigned wd = words[w];
    const __half2 lo = __floats2half2_rn(e2t_tab[wd & 0xF] * sf,
                                         e2t_tab[(wd >> 4) & 0xF] * sf);
    const __half2 hi = __floats2half2_rn(e2t_tab[(wd >> 8) & 0xF] * sf,
                                         e2t_tab[(wd >> 12) & 0xF] * sf);
    out[4 * w] = lo.x;
    out[4 * w + 1] = lo.y;
    out[4 * w + 2] = hi.x;
    out[4 * w + 3] = hi.y;
  }
}

// Causal visibility: col is the within-tile KV column (bounds the staged
// panel); kv_global = tile_start + col is checked against the per-row causal
// horizon (row token_idx sees KV [0, prefix + token_idx]).
__device__ __forceinline__ bool key_visible(
    const int col, const int token_idx, const int valid_k_rows,
    const int prefix_kv_len, const int kv_global) {
  return col < valid_k_rows && kv_global <= prefix_kv_len + token_idx;
}

template <int H>
__global__ void __launch_bounds__(kThreads, 1)
    grouped_verify_nvfp4_kernel(
        const __half* __restrict__ q, const uint8_t* __restrict__ kv_cache,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        __half* __restrict__ partial_out, float* __restrict__ partial_lse,
        const int query_len, const int max_num_blocks, const int n_page,
        const float sm_scale, const float k_scale, const float v_scale,
        const int64_t page_stride) {
  const int split_id = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int r = blockIdx.z;
  if (head_idx >= H || query_len > kMaxQ) {
    return;
  }
  const int total_kv = seq_lens[r];
  const int n_splits = active_splits(total_kv);
  if (split_id >= n_splits) {
    return;
  }
  const int HQ = H * kHeads;
  partial_out +=
      ((int64_t)r * kSplits + split_id) * kMaxQ * HQ * kHeadDim;
  partial_lse += ((int64_t)r * kSplits + split_id) * kMaxQ * HQ;

  const int total_tiles = (total_kv + kBlockN - 1) / kBlockN;
  const int base_tiles = total_tiles / n_splits;
  const int extra = total_tiles - base_tiles * n_splits;
  const int split_tile_start =
      split_id * base_tiles + min(split_id, extra);
  const int split_tiles = base_tiles + (split_id < extra ? 1 : 0);
  const int split_start = split_tile_start * kBlockN;
  const int split_end = min(total_kv, split_start + split_tiles * kBlockN);
  const int prefix_kv_len = total_kv - query_len;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  // CTA head group: KV head = blockIdx.y; Q heads [6*kv_head, 6*kv_head+6).
  const int q_head_base = head_idx * kHeads;

  extern __shared__ char smem_raw[];
  Smem& smem = *reinterpret_cast<Smem*>(smem_raw);
  __half* shared_q = smem.storage.compute.q;
  __half* shared_kv = smem.storage.compute.kv;
  float* shared_scores = smem.storage.compute.scores;
  __half* shared_probs = smem.storage.compute.probs;

  // Dequant tables (e4m3 byte * layer scale, K and V sides).
  for (int b = tid; b < 256; b += kThreads) {
    smem.scale_tab[b] = e4m3_to_float((unsigned char)b) * k_scale;
    smem.scale_tab[256 + b] = e4m3_to_float((unsigned char)b) * v_scale;
  }
  for (int b = tid; b < 16; b += kThreads) {
    smem.e2t_tab[b] = e2m1_f((unsigned)b);
  }

  // Q panel: 48 packed rows (token, head), sm_scale folded.
  {
    constexpr int kVecsPerRow = kHeadDim / 8;  // uint4 of halves
    const uint4* q_vec = reinterpret_cast<const uint4*>(q);
    uint4* shared_q_vec = reinterpret_cast<uint4*>(shared_q);
    for (int idx = tid; idx < kRows * kVecsPerRow; idx += kThreads) {
      const int row = idx / kVecsPerRow;
      const int vec_col = idx % kVecsPerRow;
      const int token_idx = row / kHeads;
      const int head = q_head_base + row % kHeads;
      const __half2 s = __float2half2_rn(sm_scale);
      uint4 v;
      if (token_idx < query_len) {
        v = __ldg(q_vec +
                  ((int64_t)(r * kMaxQ + token_idx) * H * kHeads + head) *
                      kVecsPerRow + vec_col);
        // Fold sm_scale per half2 (4 per uint4), writing straight to smem.
        const __half2* src = reinterpret_cast<const __half2*>(&v);
        __half2* dst = reinterpret_cast<__half2*>(
            shared_q + row * kQStride + vec_col * 8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          dst[j] = __hmul2(src[j], s);
        }
        continue;
      }
      v = make_uint4(0, 0, 0, 0);
      shared_q_vec[row * (kQStride / 8) + vec_col] = v;
    }
  }
  if (tid < kRows) {
    smem.row_max[tid] = kNegInf;
    smem.row_sum[tid] = 0.0f;
  }
  __syncthreads();

  fragment<accumulator, 16, 16, 16, float>
      output_fragments[kOutputTilesPerWarp];
#pragma unroll
  for (int f = 0; f < kOutputTilesPerWarp; ++f) {
    fill_fragment(output_fragments[f], 0.0f);
  }

  // Pass 1: exact per-split max/sum over this split's KV tiles (K only).
  for (int tile_start = split_start; tile_start < split_end;
       tile_start += kBlockN) {
    const int valid = min(kBlockN, split_end - tile_start);
    load_kv_panel(kv_cache, block_table, shared_kv, smem.e2t_tab,
                  smem.scale_tab, r, head_idx, H, 0 /*K side*/, tile_start,
                  valid, n_page, max_num_blocks, page_stride);
    for (int idx = tid + valid * (kKVStride / 8);
         idx < kBlockN * (kKVStride / 8); idx += kThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    grouped_verify_qk(shared_q, shared_kv, shared_scores);
    __syncthreads();

#pragma unroll
    for (int row = warp_id; row < kRows; row += kWarps) {
      const int token_idx = row / kHeads;
      const int col = lane_id;
      const int kv_idx = tile_start + col;
      const bool visible = key_visible(col, token_idx, valid, prefix_kv_len,
                                       kv_idx);
      const float score =
          visible ? shared_scores[row * kScoreStride + col] : kNegInf;
      const float tile_max = warp_reduce_max(score);
      const float probability =
          visible ? __expf(fmaxf(score - tile_max, -80.0f)) : 0.0f;
      const float tile_sum = warp_reduce_sum(probability);
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

  // Pass 2: recompute QK, form P with the exact split max/sum, P @ V.
  for (int tile_start = split_start; tile_start < split_end;
       tile_start += kBlockN) {
    const int valid = min(kBlockN, split_end - tile_start);
    load_kv_panel(kv_cache, block_table, shared_kv, smem.e2t_tab,
                  smem.scale_tab, r, head_idx, H, 0 /*K side*/, tile_start,
                  valid, n_page, max_num_blocks, page_stride);
    for (int idx = tid + valid * (kKVStride / 8);
         idx < kBlockN * (kKVStride / 8); idx += kThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    grouped_verify_qk(shared_q, shared_kv, shared_scores);
    __syncthreads();

    for (int idx = tid; idx < kRows * kBlockN; idx += kThreads) {
      const int row = idx / kBlockN;
      const int col = idx % kBlockN;
      const int token_idx = row / kHeads;
      const int kv_idx = tile_start + col;
      const bool visible =
          key_visible(col, token_idx, valid, prefix_kv_len, kv_idx) &&
          smem.row_sum[row] > 0.0f;
      const float probability =
          visible ? __expf(fmaxf(shared_scores[row * kScoreStride + col] -
                                     smem.row_max[row],
                                 -80.0f))
                  : 0.0f;
      shared_probs[row * kProbStride + col] = __float2half_rn(probability);
    }
    __syncthreads();

    load_kv_panel(kv_cache, block_table, shared_kv, smem.e2t_tab,
                  smem.scale_tab, r, head_idx, H, 1 /*V side*/, tile_start,
                  valid, n_page, max_num_blocks, page_stride);
    for (int idx = tid + valid * (kKVStride / 8);
         idx < kBlockN * (kKVStride / 8); idx += kThreads) {
      reinterpret_cast<uint4*>(shared_kv)[idx] = make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    // P already uses the exact split max/sum: plain accumulation, no
    // per-tile rescale.
#pragma unroll
    for (int f = 0; f < kOutputTilesPerWarp; ++f) {
      const int output_tile = warp_id + f * kWarps;
      const int m_tile = output_tile / (kHeadDim / 16);
      const int d_tile = output_tile % (kHeadDim / 16);
      fragment<matrix_a, 16, 16, 16, half, row_major> probability_fragment;
      fragment<matrix_b, 16, 16, 16, half, row_major> value_fragment;
      fragment<accumulator, 16, 16, 16, float> tile_fragment;
      fill_fragment(tile_fragment, 0.0f);
#pragma unroll
      for (int k_offset = 0; k_offset < kBlockN; k_offset += 16) {
        load_matrix_sync(probability_fragment,
                         shared_probs + m_tile * 16 * kProbStride +
                             k_offset,
                         kProbStride);
        load_matrix_sync(value_fragment,
                         shared_kv + k_offset * kKVStride + d_tile * 16,
                         kKVStride);
        mma_sync(tile_fragment, probability_fragment, value_fragment,
                 tile_fragment);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        output_fragments[f].x[i] += tile_fragment.x[i];
      }
    }
    __syncthreads();
  }

  // Store per-split normalized partials (out * v-independent / sum, LSE).
  __syncthreads();
  float* shared_output = smem.storage.output;
#pragma unroll
  for (int f = 0; f < kOutputTilesPerWarp; ++f) {
    const int output_tile = warp_id + f * kWarps;
    const int m_tile = output_tile / (kHeadDim / 16);
    const int d_tile = output_tile % (kHeadDim / 16);
    store_matrix_sync(
        shared_output + m_tile * 16 * kHeadDim + d_tile * 16,
        output_fragments[f], kHeadDim, mem_row_major);
  }
  __syncthreads();

  for (int idx = tid; idx < kRows * kHeadDim; idx += kThreads) {
    const int row = idx / kHeadDim;
    const int d = idx % kHeadDim;
    const int token_idx = row / kHeads;
    if (token_idx < query_len) {
      const float sum = smem.row_sum[row];
      const float scale = sum > 0.0f ? 1.0f / sum : 0.0f;
      const int head = q_head_base + row % kHeads;
      partial_out[((int64_t)token_idx * HQ + head) * kHeadDim +
                  d] = __float2half_rn(shared_output[idx] * scale);
    }
  }
  if (tid < kRows) {
    const int token_idx = tid / kHeads;
    if (token_idx < query_len) {
      const float sum = smem.row_sum[tid];
      const int head = q_head_base + tid % kHeads;
      partial_lse[(int64_t)token_idx * HQ + head] =
          sum > 0.0f ? smem.row_max[tid] + logf(sum) : kNegInf;
    }
  }
}

template <int H>
__global__ void __launch_bounds__(kThreads)
    grouped_verify_combine_kernel(
        const __half* __restrict__ partial_out,
        const float* __restrict__ partial_lse,
        const int* __restrict__ seq_lens, __half* __restrict__ out,
        const int query_len) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int r = blockIdx.z;
  if (token_idx >= query_len || head_idx >= H * 6) {
    return;
  }
  const int HQ = H * kHeads;
  partial_out += ((int64_t)r * kSplits) * kMaxQ * HQ * kHeadDim;
  partial_lse += ((int64_t)r * kSplits) * kMaxQ * HQ;
  out += (int64_t)r * query_len * HQ * kHeadDim;
  const int n_splits = active_splits(seq_lens[r]);

  __shared__ float split_lse[kSplits];
  __shared__ float final_max;
  __shared__ float final_inv_sum;
  if (threadIdx.x < kSplits) {
    const int64_t lse_idx =
        ((int64_t)threadIdx.x * kMaxQ + token_idx) * HQ + head_idx;
    split_lse[threadIdx.x] =
        threadIdx.x < n_splits ? partial_lse[lse_idx] : kNegInf;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float max_lse = kNegInf;
    for (int s = 0; s < n_splits; ++s) {
      max_lse = fmaxf(max_lse, split_lse[s]);
    }
    float sum = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      if (split_lse[s] > -1.0e20f) {
        sum += __expf(fmaxf(split_lse[s] - max_lse, -80.0f));
      }
    }
    final_max = max_lse;
    final_inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  }
  __syncthreads();

  for (int d = threadIdx.x; d < kHeadDim; d += kThreads) {
    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      if (split_lse[s] > -1.0e20f) {
        const float weight =
            __expf(fmaxf(split_lse[s] - final_max, -80.0f)) * final_inv_sum;
        const int64_t idx =
            (((int64_t)s * kMaxQ + token_idx) * HQ + head_idx) * kHeadDim + d;
        acc = fmaf(weight, __half2float(partial_out[idx]), acc);
      }
    }
    out[(token_idx * HQ + head_idx) * kHeadDim + d] = __float2half_rn(acc);
  }
}

}  // namespace sm70gv

using namespace sm70gv;

template <int H>
void grouped_verify_nvfp4_impl(torch::stable::Tensor& q,
                               torch::stable::Tensor& kv,
                               torch::stable::Tensor& block_table,
                               torch::stable::Tensor& kv_lens,
                               torch::stable::Tensor& out, double k_scale,
                               double v_scale, double sm_scale,
                               int64_t q_rows, torch::stable::Tensor& ws) {
  torch::stable::accelerator::DeviceGuard device_guard(
      q.get_device_index());
  STD_TORCH_CHECK(q.size(2) == H * kHeads && q.size(3) == kHeadDim,
                  "grouped verify: q heads/dim mismatch (HQ=", H * kHeads,
                  ")");
  STD_TORCH_CHECK(q_rows >= 1 && q_rows <= kMaxQ,
                  "grouped verify: q_rows must be in [1, 8] (MTP/DFlash "
                  "verify rows); the Q panel is sized for 8, rows past "
                  "query_len are zeroed");
  const int64_t n_page = kv.size(2);
  STD_TORCH_CHECK(kv.size(1) == 2 * H && n_page % 16 == 0 &&
                      kv.size(3) == kKVBytes,
                  "grouped verify: kv layout mismatch (pages, 2*H=", H,
                  ", page%16==0, F=144)");
  const int64_t R = q.size(0);
  const int HQ = H * kHeads;
  const int64_t bt_cols = block_table.size(1);
  STD_TORCH_CHECK(block_table.size(0) == R && kv_lens.size(0) == R,
                  "grouped verify: table/len mismatch");
  // Workspace is fp32: [partials: R*80*8*HQ*256 halves][lse: R*80*8*HQ*256
  // floats] = R*80*8*HQ*(512 + 1024) bytes.
  const int64_t ws_need = (int64_t)R * kSplits * kMaxQ * HQ * 384;
  STD_TORCH_CHECK(ws.numel() >= ws_need,
                  "grouped verify: workspace too small (need ", ws_need,
                  " floats)");

  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());
  const int smem_bytes = sizeof(Smem);
  static bool smem_set =
      (cudaFuncSetAttribute(
           grouped_verify_nvfp4_kernel<H>,
           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) ==
       cudaSuccess);
  STD_TORCH_CHECK(smem_set, "grouped verify: smem opt-in failed");

  const int64_t partial_elems =
      (int64_t)R * kSplits * kMaxQ * HQ * kHeadDim;  // halves
  __half* partial = reinterpret_cast<__half*>(ws.mutable_data_ptr());
  float* lse = reinterpret_cast<float*>(partial + partial_elems);

  dim3 grid(kSplits, H, R);
  dim3 block(kThreads);
  grouped_verify_nvfp4_kernel<H>
      <<<grid, block, smem_bytes, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr()),
          reinterpret_cast<const uint8_t*>(kv.data_ptr()),
          reinterpret_cast<const int*>(block_table.data_ptr()),
          reinterpret_cast<const int*>(kv_lens.data_ptr()), partial, lse,
          (int)q_rows, (int)bt_cols, (int)n_page,
          (float)sm_scale, (float)k_scale, (float)v_scale, kv.stride(0));

  dim3 cgrid(kMaxQ, H * 6, R);
  grouped_verify_combine_kernel<H>
      <<<cgrid, block, 0, stream>>>(
          partial, lse, reinterpret_cast<const int*>(kv_lens.data_ptr()),
          reinterpret_cast<__half*>(out.mutable_data_ptr()), (int)q_rows);

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "grouped verify launch failed: ",
                  cudaGetErrorString(err));
}

void sm70_grouped_verify_nvfp4(torch::stable::Tensor& q,
                               torch::stable::Tensor& kv,
                               torch::stable::Tensor& block_table,
                               torch::stable::Tensor& kv_lens,
                               torch::stable::Tensor& out, double k_scale,
                               double v_scale, double sm_scale,
                               int64_t q_rows, torch::stable::Tensor& ws) {
  const int64_t H = kv.size(1) / 2;
  const int64_t HQ = q.size(2);
  if (H == 1 && HQ == 6) {
    grouped_verify_nvfp4_impl<1>(q, kv, block_table, kv_lens, out, k_scale,
                                 v_scale, sm_scale, q_rows, ws);
  } else if (H == 2 && HQ == 12) {
    grouped_verify_nvfp4_impl<2>(q, kv, block_table, kv_lens, out, k_scale,
                                 v_scale, sm_scale, q_rows, ws);
  } else {
    STD_TORCH_CHECK(false, "grouped verify: supports (H,HQ)=", H, ",", HQ,
                    " (TP4 (1,6) or TP2 (2,12)), D=256");
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_gv_ops) {
  sm70_gv_ops.def(
      "sm70_grouped_verify_nvfp4(Tensor q, Tensor kv_cache, "
      "Tensor block_table, Tensor kv_lens, Tensor out, float k_scale, "
      "float v_scale, float sm_scale, int q_rows, Tensor workspace) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_gv_ops) {
  sm70_gv_ops.impl(
      "sm70_grouped_verify_nvfp4", TORCH_BOX(&::sm70_grouped_verify_nvfp4));
}
