// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
#define VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <cstdlib>

namespace vllm::qsa {

constexpr int kLexicographicTopKThreads = 1024;
constexpr int kLexicographicTopKBins = 256;
constexpr int kLexicographicTopKDecodeCandidateCapacity = 2304;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  // IEEE -0.0 and +0.0 compare equal, so keep them in the same score bucket
  // and let the block index provide the deterministic tie break.
  if (value == 0.0f) value = 0.0f;
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int TopK>
struct LexicographicTopKShared {
  using BlockScan = cub::BlockScan<uint64_t, kLexicographicTopKThreads>;

  uint32_t histogram[kLexicographicTopKBins];
  uint32_t warp_tot[8];
  typename BlockScan::TempStorage scan;
  uint32_t prefix;
  uint32_t pivot;
  uint32_t pivot_incl;
  uint32_t remaining;
  uint32_t greater_seen;
  uint32_t equal_seen;
  uint32_t chunk_greater_base;
  uint32_t chunk_equal_base;
};

template <int TopK>
struct LexicographicDecodeTopKShared {
  using BlockScan = cub::BlockScan<uint64_t, kLexicographicTopKThreads>;

  // The decode fast path first selects one coarse radix bucket, then scans
  // only that bucket for the remaining bytes. Keep two buffers so compaction
  // never overwrites input indices that another warp has not consumed yet.
  uint32_t histogram[2][kLexicographicTopKBins + 128];
  int32_t candidates[2][kLexicographicTopKDecodeCandidateCapacity];
  typename BlockScan::TempStorage scan;
  uint32_t prefix;
  uint32_t pivot;
  uint32_t remaining;
  uint32_t remaining_ties;
  uint32_t candidate_count[2];
  uint32_t threshold_bin;
  uint32_t greater_seen;
  uint32_t equal_seen;
  uint32_t chunk_greater_base;
  uint32_t chunk_equal_base;
};

template <int TopK>
__device__ __forceinline__ void decode_suffix_scan_histogram(
    LexicographicDecodeTopKShared<TopK>& shared) {
#pragma unroll
  for (int pass = 0; pass < 8; ++pass) {
    const int distance = 1 << pass;
    const int source = pass & 1;
    if (threadIdx.x < kLexicographicTopKBins) {
      uint32_t value = shared.histogram[source][threadIdx.x];
      if (threadIdx.x + distance < kLexicographicTopKBins) {
        value += shared.histogram[source][threadIdx.x + distance];
      }
      shared.histogram[source ^ 1][threadIdx.x] = value;
    }
    __syncthreads();
  }
}

template <int TopK>
__device__ __forceinline__ void decode_choose_threshold(
    LexicographicDecodeTopKShared<TopK>& shared, int shift) {
  if (threadIdx.x < kLexicographicTopKBins &&
      shared.histogram[0][threadIdx.x] > shared.remaining &&
      shared.histogram[0][threadIdx.x + 1] <= shared.remaining) {
    shared.threshold_bin = threadIdx.x;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t bin = shared.threshold_bin;
    const uint32_t greater = shared.histogram[0][bin + 1];
    shared.remaining -= greater;
    shared.prefix |= bin << shift;
    if (shared.remaining == 0) {
      const uint32_t low_mask = shift == 0 ? 0u : ((uint32_t{1} << shift) - 1);
      shared.pivot = shared.prefix | low_mask;
      shared.remaining_ties = 0;
    } else if (shift == 0) {
      shared.pivot = shared.prefix;
      shared.remaining_ties = shared.remaining;
    }
  }
  __syncthreads();
}

template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_topk_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t num_rows, uint32_t columns,
    uint32_t stride) {
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t raw_length = lengths[row];
  const uint32_t length =
      raw_length > 0 ? min(static_cast<uint32_t>(raw_length), columns) : 0;
  const float* row_logits = logits + static_cast<uint64_t>(row) * stride;
  int32_t* row_output = output + static_cast<uint64_t>(row) * TopK;

  if (length <= TopK) {
    for (uint32_t index = tx; index < TopK;
         index += kLexicographicTopKThreads) {
      row_output[index] = index < length ? static_cast<int32_t>(index) : -1;
    }
    return;
  }

  __shared__ LexicographicTopKShared<TopK> shared;
  if (tx == 0) {
    shared.prefix = 0;
    shared.remaining = TopK;
  }
  __syncthreads();

  // Select the exact k-th score with four byte-wide radix passes. The prior
  // implementation also radix-selected the 32-bit index tie-break, requiring
  // eight full scans. We only need the score pivot: the final increasing-index
  // compaction can admit exactly `remaining` values from the pivot bucket.
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    for (uint32_t bin = tx; bin < kLexicographicTopKBins;
         bin += kLexicographicTopKThreads) {
      shared.histogram[bin] = 0;
    }
    __syncthreads();

    const int shift = 24 - pass * 8;
    const uint32_t prefix = shared.prefix;
    const uint32_t prefix_mask = pass == 0 ? 0 : (~uint32_t{0} << (shift + 8));
    for (uint32_t index = tx; index < length;
         index += kLexicographicTopKThreads) {
      const uint32_t key = ordered_float_bits(row_logits[index]);
      if ((key & prefix_mask) == prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();

    // Parallel pivot selection (replaces the thread-0 256-step serial bin
    // walk, which measured 6-40% of kernel time): warp-scan the 256-bin
    // histogram and pick the unique crossing bin b with
    // incl[b-1] <= total - remaining < incl[b], i.e. G(b) < remaining where
    // G(b) is the count of bins above b - the walk's stopping condition.
    // Width-gated: at width <= 8192 the walk dominates (40%); at 16384 the
    // extra per-pass syncs cost more than the walk (measured), so wide rows
    // keep the serial path.
    if (columns <= 8192u)
    {
      const int warp = tx >> 5;
      const int lane = tx & 31;
      uint32_t incl = (tx < kLexicographicTopKBins) ? shared.histogram[tx] : 0u;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        // shfl.up clamps to SELF for lane < d, so only lanes >= d may add.
        const uint32_t below = __shfl_up_sync(~0u, incl, d);
        if (lane >= d) incl += below;
      }
      if (lane == 31) shared.warp_tot[warp] = incl;
      __syncthreads();
      uint32_t base = 0;
      uint32_t total = 0;
#pragma unroll
      for (int x = 0; x < 8; ++x) {
        total += shared.warp_tot[x];
        if (x < warp) base += shared.warp_tot[x];
      }
      incl += base;
      const uint32_t want = total - shared.remaining;
      const uint32_t excl =
          incl - ((tx < kLexicographicTopKBins) ? shared.histogram[tx] : 0u);
      const uint32_t cand = (tx < kLexicographicTopKBins && incl > want &&
                             excl <= want)
                                ? tx
                                : 0xffffu;
      uint32_t m = cand;
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        m = min(m, __shfl_xor_sync(~0u, m, d));
      }
      if (lane == 0) shared.warp_tot[warp] = m;
      __syncthreads();
      if (tx == 0) {
        uint32_t b = shared.warp_tot[0];
#pragma unroll
        for (int x = 1; x < 8; ++x) b = min(b, shared.warp_tot[x]);
        shared.pivot = b;
      }
      __syncthreads();
      const uint32_t bin = shared.pivot;
      if (tx == bin && bin < kLexicographicTopKBins) shared.pivot_incl = incl;
      __syncthreads();
      if (tx == 0) {
        shared.prefix |= static_cast<uint32_t>(bin) << shift;
        // bins strictly above bin = total - incl[bin]
        shared.remaining -= total - shared.pivot_incl;
      }
    }
    else if (tx == 0) {
      uint32_t remaining = shared.remaining;
      for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
        const uint32_t count = shared.histogram[bin];
        if (remaining > count) {
          remaining -= count;
        } else {
          shared.prefix |= static_cast<uint32_t>(bin) << shift;
          shared.remaining = remaining;
          break;
        }
      }
    }
    __syncthreads();
  }

  if (tx == 0) {
    shared.pivot = shared.prefix;
    shared.greater_seen = 0;
    shared.equal_seen = 0;
  }
  __syncthreads();

  // Compact in increasing index order. A packed 64-bit scan tracks counts of
  // greater and pivot-equal scores at once. Every greater score is selected;
  // only the first `remaining` pivot ties are admitted, preserving the exact
  // lower-index tie break and QSA's canonical accumulation order.
  using BlockScan = typename LexicographicTopKShared<TopK>::BlockScan;
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    const uint32_t key =
        index < length ? ordered_float_bits(row_logits[index]) : 0;
    const uint32_t greater = index < length && key > shared.pivot ? 1u : 0u;
    const uint32_t equal = index < length && key == shared.pivot ? 1u : 0u;
    const uint64_t counts = (static_cast<uint64_t>(greater) << 32) | equal;
    uint64_t prefix_counts = 0;
    uint64_t aggregate_counts = 0;
    BlockScan(shared.scan)
        .ExclusiveSum(counts, prefix_counts, aggregate_counts);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_greater_base = shared.greater_seen;
      shared.chunk_equal_base = shared.equal_seen;
      shared.greater_seen += static_cast<uint32_t>(aggregate_counts >> 32);
      shared.equal_seen += static_cast<uint32_t>(aggregate_counts);
    }
    __syncthreads();
    const uint32_t greater_before =
        shared.chunk_greater_base + static_cast<uint32_t>(prefix_counts >> 32);
    const uint32_t equal_before =
        shared.chunk_equal_base + static_cast<uint32_t>(prefix_counts);
    const bool selected = greater || (equal && equal_before < shared.remaining);
    if (selected) {
      const uint32_t offset =
          greater_before + min(equal_before, shared.remaining);
      row_output[offset] = static_cast<int32_t>(index);
    }
    __syncthreads();
  }
}

constexpr int kFastBucketCapacity = 704;
constexpr int kRelBins = 1024;

// Three-pass EXACT selector for wide rows, robust to QSA's heavily clustered
// scores (sigma ~1e-3 makes every absolute-bit radix prefix collapse to one
// bucket). Bucketing is RELATIVE to the row's [min,max] and, crucially,
// DETERMINISTIC: the bin is an integer shift, not a float multiply, so pass 2
// and pass 3 can never disagree by an ulp about an element's bin.
//   pass 1: ordered-key min/max.
//   pass 2: kRelBins histogram of (key-min) >> shift.
//   pass 3: emit every element above the pivot bin (all of them are selected)
//           and collect the pivot bin's candidates in shared.
// The exact 32-bit pivot and the increasing-index tie order are then resolved
// among those candidates, so the result is bit-identical to the baseline.
// A pivot bin wider than the capacity falls back to the baseline 4-pass radix.
template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_topk_fast_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t num_rows, uint32_t columns,
    uint32_t stride) {
  const uint32_t tx = threadIdx.x;
  const uint32_t row = blockIdx.x;
  if (row >= num_rows) return;
  using Shared = LexicographicTopKShared<TopK>;
  using BlockScan = typename Shared::BlockScan;

  const int32_t raw_length = lengths[row];
  const uint32_t length =
      raw_length > 0 ? min(static_cast<uint32_t>(raw_length), columns) : 0;
  const float* row_logits = logits + static_cast<uint64_t>(row) * stride;
  int32_t* row_output = output + static_cast<uint64_t>(row) * TopK;

  if (length <= TopK) {
    for (uint32_t index = tx; index < TopK;
         index += kLexicographicTopKThreads) {
      row_output[index] = index < length ? static_cast<int32_t>(index) : -1;
    }
    return;
  }

  __shared__ Shared shared;
  __shared__ uint32_t rel_bins[kRelBins];
  __shared__ uint32_t bucket_key[kFastBucketCapacity];
  __shared__ uint32_t bucket_idx[kFastBucketCapacity];
  __shared__ uint32_t bucket_count;
  __shared__ uint32_t key_min;
  __shared__ uint32_t key_max;
  __shared__ uint32_t bin_shift;
  __shared__ uint32_t pivot_key;
  __shared__ uint32_t above_in_bucket;
  __shared__ int pivot_bin;

  if (tx == 0) {
    shared.greater_seen = 0;
    shared.equal_seen = 0;
    shared.remaining = TopK;
    shared.prefix = 0;
    bucket_count = 0;
    key_min = 0xffffffffu;
    key_max = 0u;
    pivot_key = 0xffffffffu;
    above_in_bucket = 0;
    pivot_bin = 0;
    bin_shift = 0;
  }
  __syncthreads();

  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    if (index < length) {
      const uint32_t key = ordered_float_bits(row_logits[index]);
      atomicMin(&key_min, key);
      atomicMax(&key_max, key);
    }
  }
  __syncthreads();
  if (tx == 0) {
    // Smallest shift that maps the row's key span into kRelBins bins. Integer
    // only, so both scanning passes classify every element identically.
    uint32_t shift = 0;
    while ((uint64_t)(key_max - key_min) >> shift >= (uint64_t)kRelBins) {
      ++shift;
    }
    bin_shift = shift;
  }
  __syncthreads();

  for (uint32_t b = tx; b < kRelBins; b += kLexicographicTopKThreads) {
    rel_bins[b] = 0u;
  }
  __syncthreads();
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    if (index < length) {
      const uint32_t key = ordered_float_bits(row_logits[index]);
      const uint32_t bin = (key - key_min) >> bin_shift;
      atomicAdd(&rel_bins[bin < (uint32_t)kRelBins ? bin
                                                   : (uint32_t)kRelBins - 1u],
                1u);
    }
  }
  __syncthreads();
  if (tx == 0) {
    uint32_t remaining = shared.remaining;
    int chosen = 0;
    for (int b = kRelBins - 1; b >= 0; --b) {
      const uint32_t count = rel_bins[b];
      if (remaining > count) {
        remaining -= count;
      } else {
        chosen = b;
        shared.remaining = remaining;
        break;
      }
    }
    pivot_bin = chosen;
  }
  __syncthreads();

  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    uint32_t above = 0;
    bool in_bin = false;
    uint32_t key = 0u;
    if (index < length) {
      key = ordered_float_bits(row_logits[index]);
      const uint32_t bin = (key - key_min) >> bin_shift;
      above = bin > (uint32_t)pivot_bin ? 1u : 0u;
      in_bin = bin == (uint32_t)pivot_bin;
    }
    uint64_t prefix_counts = 0;
    uint64_t aggregate_counts = 0;
    BlockScan(shared.scan)
        .ExclusiveSum(static_cast<uint64_t>(above), prefix_counts,
                      aggregate_counts);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_greater_base = shared.greater_seen;
      shared.greater_seen += static_cast<uint32_t>(aggregate_counts);
    }
    __syncthreads();
    if (above) {
      row_output[shared.chunk_greater_base +
                 static_cast<uint32_t>(prefix_counts)] =
          static_cast<int32_t>(index);
    }
    if (in_bin) {
      const uint32_t slot = atomicAdd(&bucket_count, 1u);
      if (slot < kFastBucketCapacity) {
        bucket_key[slot] = key;
        bucket_idx[slot] = index;
      }
    }
    __syncthreads();
  }

  const uint32_t nb = bucket_count;
  const uint32_t ties = shared.remaining;
  if (nb <= kFastBucketCapacity && nb > 0 && ties > 0) {
    if (tx < nb) {
      uint32_t greater = 0;
      uint32_t eqg = 0;
      for (uint32_t other = 0; other < nb; ++other) {
        const uint32_t k = bucket_key[other];
        if (k > bucket_key[tx]) ++greater;
        if (k >= bucket_key[tx]) ++eqg;
      }
      if (greater < ties && eqg >= ties) pivot_key = bucket_key[tx];
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t above = 0;
      for (uint32_t other = 0; other < nb; ++other) {
        if (bucket_key[other] > pivot_key) ++above;
      }
      above_in_bucket = above;
    }
    __syncthreads();
    if (tx < nb && pivot_key != 0xffffffffu) {
      const uint32_t my_index = bucket_idx[tx];
      uint32_t greater_rank = 0;
      uint32_t equal_rank = 0;
      for (uint32_t other = 0; other < nb; ++other) {
        if (bucket_idx[other] >= my_index) continue;
        const uint32_t k = bucket_key[other];
        if (k > pivot_key) {
          ++greater_rank;
        } else if (k == pivot_key) {
          ++equal_rank;
        }
      }
      if (bucket_key[tx] > pivot_key) {
        row_output[shared.greater_seen + greater_rank] =
            static_cast<int32_t>(my_index);
      } else if (bucket_key[tx] == pivot_key && equal_rank < ties) {
        row_output[shared.greater_seen + above_in_bucket + equal_rank] =
            static_cast<int32_t>(my_index);
      }
    }
    return;
  }

  // Fallback: baseline 4-pass radix + emit over the whole row.
  if (tx == 0) {
    shared.greater_seen = 0;
    shared.equal_seen = 0;
    shared.remaining = TopK;
    shared.prefix = 0;
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    for (uint32_t bin = tx; bin < kLexicographicTopKBins;
         bin += kLexicographicTopKThreads) {
      shared.histogram[bin] = 0;
    }
    __syncthreads();
    const int shift = 24 - pass * 8;
    const uint32_t prefix = shared.prefix;
    const uint32_t prefix_mask = pass == 0 ? 0 : (~uint32_t{0} << (shift + 8));
    for (uint32_t index = tx; index < length;
         index += kLexicographicTopKThreads) {
      const uint32_t key = ordered_float_bits(row_logits[index]);
      if ((key & prefix_mask) == prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t remaining = shared.remaining;
      for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
        const uint32_t count = shared.histogram[bin];
        if (remaining > count) {
          remaining -= count;
        } else {
          shared.prefix |= static_cast<uint32_t>(bin) << shift;
          shared.remaining = remaining;
          break;
        }
      }
    }
    __syncthreads();
  }
  if (tx == 0) shared.pivot = shared.prefix;
  __syncthreads();
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    const uint32_t key =
        index < length ? ordered_float_bits(row_logits[index]) : 0u;
    const uint32_t greater = (index < length && key > shared.pivot) ? 1u : 0u;
    const uint32_t equal = (index < length && key == shared.pivot) ? 1u : 0u;
    const uint64_t counts = (static_cast<uint64_t>(greater) << 32) | equal;
    uint64_t prefix_counts = 0;
    uint64_t aggregate_counts = 0;
    BlockScan(shared.scan)
        .ExclusiveSum(counts, prefix_counts, aggregate_counts);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_greater_base = shared.greater_seen;
      shared.chunk_equal_base = shared.equal_seen;
      shared.greater_seen += static_cast<uint32_t>(aggregate_counts >> 32);
      shared.equal_seen += static_cast<uint32_t>(aggregate_counts);
    }
    __syncthreads();
    const uint32_t greater_before =
        shared.chunk_greater_base + static_cast<uint32_t>(prefix_counts >> 32);
    const uint32_t equal_before =
        shared.chunk_equal_base + static_cast<uint32_t>(prefix_counts);
    const bool selected = greater || (equal && equal_before < shared.remaining);
    if (selected) {
      row_output[greater_before + min(equal_before, shared.remaining)] =
          static_cast<int32_t>(index);
    }
    __syncthreads();
  }
}

// Single-token QSA decode has only about two thousand live block scores at the
// common 8K context length. After the first radix byte, scanning all scores for
// the other three bytes wastes most of the work. Compact the selected coarse
// bucket into shared memory and refine that much smaller set instead. Integer
// counters and the final increasing-index pass retain exact tie-breaking.
template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_decode_topk_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t columns) {
  const uint32_t tx = threadIdx.x;
  const int32_t raw_length = lengths[0];
  const uint32_t length =
      raw_length > 0 ? min(static_cast<uint32_t>(raw_length), columns) : 0;

  if (length <= TopK) {
    for (uint32_t index = tx; index < TopK;
         index += kLexicographicTopKThreads) {
      output[index] = index < length ? static_cast<int32_t>(index) : -1;
    }
    return;
  }

  __shared__ LexicographicDecodeTopKShared<TopK> shared;
  if (tx == 0) {
    shared.prefix = 0;
    shared.remaining = TopK;
    shared.remaining_ties = 0;
    shared.candidate_count[0] = 0;
  }
  __syncthreads();

  // One path for every row length: the first 8-bit pass narrows the row to a
  // coarse bucket, which is what must fit in shared memory (not the row), so
  // long rows get the compacting path too. A bucket wider than the candidate
  // buffer falls back to the exact four-pass scan below.
    // Coarse pass over the complete score row.
    if (tx == 0) shared.chunk_greater_base = 0u;  // 1 = coarse bucket overflow
    if (tx < kLexicographicTopKBins + 1) shared.histogram[0][tx] = 0;
    __syncthreads();
    for (uint32_t index = tx; index < length;
         index += kLexicographicTopKThreads) {
      const uint32_t key = ordered_float_bits(logits[index]);
      atomicAdd(&shared.histogram[0][key >> 24], 1u);
    }
    __syncthreads();
    decode_suffix_scan_histogram(shared);
    decode_choose_threshold(shared, 24);

    if (shared.remaining != 0) {
      if (tx < kLexicographicTopKBins + 1) shared.histogram[0][tx] = 0;
      __syncthreads();
      for (uint32_t index = tx; index < length;
           index += kLexicographicTopKThreads) {
        const uint32_t key = ordered_float_bits(logits[index]);
        if ((key & 0xff000000u) == shared.prefix) {
          const uint32_t position = atomicAdd(&shared.candidate_count[0], 1u);
          if (position < static_cast<uint32_t>(
                             kLexicographicTopKDecodeCandidateCapacity)) {
            shared.candidates[0][position] = static_cast<int32_t>(index);
            atomicAdd(&shared.histogram[0][(key >> 16) & 0xffu], 1u);
          } else {
            shared.chunk_greater_base = 1u;  // bucket too wide: exact fallback
          }
        }
      }
      __syncthreads();
    }

    if (shared.chunk_greater_base == 0u) {
#pragma unroll
    for (int radix_pass = 0; radix_pass < 3; ++radix_pass) {
      if (shared.remaining == 0) break;
      const int shift = 16 - radix_pass * 8;
      decode_suffix_scan_histogram(shared);
      decode_choose_threshold(shared, shift);
      if (shared.remaining == 0 || shift == 0) break;

      const int source = radix_pass & 1;
      const int target = source ^ 1;
      if (tx == 0) shared.candidate_count[target] = 0;
      if (tx < kLexicographicTopKBins + 1) shared.histogram[0][tx] = 0;
      __syncthreads();
      const uint32_t count = shared.candidate_count[source];
      const uint32_t prefix_mask = ~uint32_t{0} << shift;
      const int next_shift = shift - 8;
      for (uint32_t item = tx; item < count;
           item += kLexicographicTopKThreads) {
        const int32_t index = shared.candidates[source][item];
        const uint32_t key = ordered_float_bits(logits[index]);
        if ((key & prefix_mask) == shared.prefix) {
          const uint32_t position =
              atomicAdd(&shared.candidate_count[target], 1u);
          shared.candidates[target][position] = index;
          atomicAdd(&shared.histogram[0][(key >> next_shift) & 0xffu], 1u);
        }
      }
      __syncthreads();
    }
    }

  if (shared.chunk_greater_base == 1u) {
    // Exact fallback: the original four-pass scan over the whole row.
    if (tx == 0) {
      shared.prefix = 0;
      shared.remaining = TopK;
      shared.remaining_ties = 0;
    }
    __syncthreads();
    // Preserve the original exact four-pass algorithm for long contexts,
    // without a host synchronization or a second kernel launch.
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      for (uint32_t bin = tx; bin < kLexicographicTopKBins;
           bin += kLexicographicTopKThreads) {
        shared.histogram[0][bin] = 0;
      }
      __syncthreads();

      const int shift = 24 - pass * 8;
      const uint32_t prefix = shared.prefix;
      const uint32_t prefix_mask =
          pass == 0 ? 0 : (~uint32_t{0} << (shift + 8));
      for (uint32_t index = tx; index < length;
           index += kLexicographicTopKThreads) {
        const uint32_t key = ordered_float_bits(logits[index]);
        if ((key & prefix_mask) == prefix) {
          atomicAdd(&shared.histogram[0][(key >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();

      if (tx == 0) {
        uint32_t remaining = shared.remaining;
        for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
          const uint32_t count = shared.histogram[0][bin];
          if (remaining > count) {
            remaining -= count;
          } else {
            shared.prefix |= static_cast<uint32_t>(bin) << shift;
            shared.remaining = remaining;
            break;
          }
        }
      }
      __syncthreads();
    }
    if (tx == 0) {
      shared.pivot = shared.prefix;
      shared.remaining_ties = shared.remaining;
    }
    __syncthreads();
  }

  if (tx == 0) {
    shared.greater_seen = 0;
    shared.equal_seen = 0;
  }
  __syncthreads();

  // Emit in original index order, matching QSA's canonical accumulation order.
  using BlockScan = typename LexicographicDecodeTopKShared<TopK>::BlockScan;
  // Each thread owns EPT consecutive indices, so one cross-thread scan covers
  // EPT*threads elements instead of one scan per 1024 elements. On a 65K row
  // that is 16 chunk scans instead of 64, and the per-chunk barrier was what
  // dominated the emit (measured 208 -> 154 us when the compacting path landed,
  // with the emit still taking most of it).
  constexpr int EPT = 8;
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads * EPT) {
    uint32_t g_loc[EPT];
    uint32_t e_loc[EPT];
    uint32_t idx[EPT];
    uint32_t g_cnt = 0, e_cnt = 0;
#pragma unroll
    for (int k = 0; k < EPT; ++k) {
      const uint32_t index = base + tx * EPT + k;
      idx[k] = index;
      const uint32_t key = index < length ? ordered_float_bits(logits[index]) : 0;
      g_loc[k] = (index < length && key > shared.pivot) ? 1u : 0u;
      e_loc[k] = (index < length && key == shared.pivot) ? 1u : 0u;
      g_cnt += g_loc[k];
      e_cnt += e_loc[k];
    }
    const uint64_t counts = (static_cast<uint64_t>(g_cnt) << 32) | e_cnt;
    uint64_t prefix_counts = 0;
    uint64_t aggregate_counts = 0;
    BlockScan(shared.scan)
        .ExclusiveSum(counts, prefix_counts, aggregate_counts);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_greater_base = shared.greater_seen;
      shared.chunk_equal_base = shared.equal_seen;
      shared.greater_seen += static_cast<uint32_t>(aggregate_counts >> 32);
      shared.equal_seen += static_cast<uint32_t>(aggregate_counts);
    }
    __syncthreads();
    uint32_t greater_before =
        shared.chunk_greater_base + static_cast<uint32_t>(prefix_counts >> 32);
    uint32_t equal_before =
        shared.chunk_equal_base + static_cast<uint32_t>(prefix_counts);
#pragma unroll
    for (int k = 0; k < EPT; ++k) {
      const bool selected =
          g_loc[k] || (e_loc[k] && equal_before < shared.remaining_ties);
      if (selected) {
        const uint32_t offset =
            greater_before + min(equal_before, shared.remaining_ties);
        output[offset] = static_cast<int32_t>(idx[k]);
      }
      greater_before += g_loc[k];
      equal_before += e_loc[k];
    }
    __syncthreads();
  }
}

template <int TopK>
void launch_qsa_lexicographic_topk(const float* logits, const int32_t* lengths,
                                   int32_t* output, uint32_t num_rows,
                                   uint32_t columns, uint32_t stride,
                                   cudaStream_t stream) {
  if (num_rows == 1) {
    qsa_lexicographic_decode_topk_kernel<TopK>
        <<<1, kLexicographicTopKThreads, 0, stream>>>(logits, lengths, output,
                                                      columns);
  } else if (columns >= 2048u) {
    // Wide rows: the relative-bucketing selector scans the score row twice
    // instead of four times (plus the emit), which is where long-context
    // prefill spends its indexer time.
    qsa_lexicographic_topk_fast_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, num_rows, columns, stride);
  } else {
    qsa_lexicographic_topk_kernel<TopK>
        <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
            logits, lengths, output, num_rows, columns, stride);
  }
}

}  // namespace vllm::qsa

#endif  // VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
