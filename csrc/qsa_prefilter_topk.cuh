// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the VLLM project
//
// Exact tile-max prefilter for the QSA indexer selection.
//
// The direct selector scans the whole score row three to four times. On a
// 500K prefill that is 20.3% of device time in the wide region, and the rows
// are clustered (sigma ~1e-3) so the traffic buys nothing. This prefilter
// shrinks the candidate set with a provable bound instead:
//
//   tile = 16 consecutive columns, S = the true top-K set, e_min = min(S).
//   A tile holding an element of S has max >= e_min; a tile holding none has
//   every element < e_min, hence max < e_min. So the K tiles with the largest
//   max contain every tile that carries an element of S, i.e. all of S.
//
// Both selections then run the exact four-pass radix kernel
// (qsa_lexicographic_topk_kernel) on the small arrays, so the result is exact
// by construction -- no bucketing heuristics. The tile ids are sorted
// ascending before the gather so a candidate's position is monotone in its
// original column, which keeps the (score desc, index asc) tie break
// bit-identical to a direct scan.

#ifndef VLLM_CSRC_QSA_PREFILTER_TOPK_CUH_
#define VLLM_CSRC_QSA_PREFILTER_TOPK_CUH_

#include <cuda_runtime.h>
#include <cstdint>

#include "qsa_lexicographic_topk.cuh"

namespace vllm::qsa {

// tile_lengths[row] = ceil(lengths[row] / tile); span_lengths[row] = span.
__global__ void qsa_prefilter_lengths_kernel(
    const int32_t* __restrict__ lengths, int32_t* __restrict__ tile_lengths,
    int32_t* __restrict__ span_lengths, uint32_t rows, uint32_t tile,
    uint32_t span) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int32_t raw = lengths[row];
  const uint32_t len = raw > 0 ? static_cast<uint32_t>(raw) : 0u;
  tile_lengths[row] = static_cast<int32_t>((len + tile - 1u) / tile);
  span_lengths[row] = static_cast<int32_t>(span);
}

// Bitonic sort of TopK int32 in shared memory, one block per row.
template <int TopK>
__global__ void qsa_prefilter_sort_kernel(int32_t* __restrict__ ids,
                                          uint32_t rows) {
  __shared__ int32_t shared[TopK];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  int32_t* base = ids + static_cast<uint64_t>(row) * TopK;
  for (uint32_t i = threadIdx.x; i < TopK; i += blockDim.x) shared[i] = base[i];
  __syncthreads();
  for (uint32_t size = 2; size <= TopK; size <<= 1) {
    for (uint32_t stride = size >> 1; stride > 0; stride >>= 1) {
      for (uint32_t i = threadIdx.x; i < TopK; i += blockDim.x) {
        const uint32_t partner = i ^ stride;
        if (partner > i) {
          const bool ascending = ((i & size) == 0);
          const int32_t a = shared[i];
          const int32_t b = shared[partner];
          if ((a > b) == ascending) {
            shared[i] = b;
            shared[partner] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  for (uint32_t i = threadIdx.x; i < TopK; i += blockDim.x) base[i] = shared[i];
}

// candidate[row, j] = logits[row, tiles[row, j / tile] * tile + j % tile],
// or -inf when the column is beyond the row's visible length. A negative tile
// id (a padded entry of the tile selection) also yields -inf.
template <int TopK>
__global__ void qsa_prefilter_gather_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    const int32_t* __restrict__ tiles, float* __restrict__ candidates,
    uint32_t rows, uint32_t stride, uint32_t tile, uint32_t span) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const int32_t raw = lengths[row];
  const uint32_t len = raw > 0 ? static_cast<uint32_t>(raw) : 0u;
  const float* __restrict__ rl = logits + static_cast<uint64_t>(row) * stride;
  const int32_t* __restrict__ tl = tiles + static_cast<uint64_t>(row) * TopK;
  float* __restrict__ out = candidates + static_cast<uint64_t>(row) * span;
  for (uint32_t j = threadIdx.x; j < span; j += blockDim.x) {
    const int32_t cell = tl[j / tile];
    const uint32_t column =
        cell < 0 ? 0xffffffffu
                 : static_cast<uint32_t>(cell) * tile + (j % tile);
    out[j] = column < len ? rl[column] : -INFINITY;
  }
}

template <int TopK>
__global__ void qsa_prefilter_map_kernel(const int32_t* __restrict__ tiles,
                                         const int32_t* __restrict__ positions,
                                         const int32_t* __restrict__ lengths,
                                         int32_t* __restrict__ output,
                                         uint32_t rows, uint32_t tile) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const int32_t raw = lengths[row];
  const uint32_t len = raw > 0 ? static_cast<uint32_t>(raw) : 0u;
  if (len <= static_cast<uint32_t>(TopK)) {
    // A row with no more visible entries than k selects all of them; the
    // prefilter has nothing to narrow down.
    for (uint32_t i = threadIdx.x; i < TopK; i += blockDim.x) {
      output[static_cast<uint64_t>(row) * TopK + i] =
          i < len ? static_cast<int32_t>(i) : -1;
    }
    return;
  }
  const int32_t* __restrict__ tl = tiles + static_cast<uint64_t>(row) * TopK;
  for (uint32_t i = threadIdx.x; i < TopK; i += blockDim.x) {
    const int32_t position =
        positions[static_cast<uint64_t>(row) * TopK + i];
    output[static_cast<uint64_t>(row) * TopK + i] =
        position < 0
            ? -1
            : tl[position / static_cast<int32_t>(tile)] *
                      static_cast<int32_t>(tile) +
                  (position % static_cast<int32_t>(tile));
  }
}

template <int TopK>
void launch_qsa_prefilter_topk(const float* logits, const float* tile_max,
                               const int32_t* lengths, int32_t* tile_lengths,
                               int32_t* span_lengths, int32_t* tiles,
                               float* candidates, int32_t* positions,
                               int32_t* output, uint32_t num_rows,
                               uint32_t columns, uint32_t num_tiles,
                               uint32_t tile, cudaStream_t stream) {
  constexpr uint32_t kThreads = 1024;
  const uint32_t span = static_cast<uint32_t>(TopK) * tile;
  qsa_prefilter_lengths_kernel<<<(num_rows + 255) / 256, 256, 0, stream>>>(
      lengths, tile_lengths, span_lengths, num_rows, tile, span);
  // Stage 1: exact top-K over the per-tile maxima.
  qsa_lexicographic_topk_kernel<TopK>
      <<<num_rows, kThreads, 0, stream>>>(tile_max, tile_lengths, tiles,
                                         num_rows, num_tiles, num_tiles);
  // Ascending tile ids make a candidate's position monotone in its column.
  qsa_prefilter_sort_kernel<TopK><<<num_rows, 512, 0, stream>>>(tiles, num_rows);
  qsa_prefilter_gather_kernel<TopK><<<num_rows, kThreads, 0, stream>>>(
      logits, lengths, tiles, candidates, num_rows, columns, tile, span);
  // Stage 2: exact top-K over the gathered candidates.
  qsa_lexicographic_topk_kernel<TopK><<<num_rows, kThreads, 0, stream>>>(
      candidates, span_lengths, positions, num_rows, span, span);
  qsa_prefilter_map_kernel<TopK><<<num_rows, 256, 0, stream>>>(
      tiles, positions, lengths, output, num_rows, tile);
}

}  // namespace vllm::qsa

#endif  // VLLM_CSRC_QSA_PREFILTER_TOPK_CUH_
