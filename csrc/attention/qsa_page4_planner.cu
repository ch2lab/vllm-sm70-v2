// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// In-tree QSA page4 planner: maps grouped four-token sparse selections to
// physical microblocks for the page4 verify path (grouped sparse page4).
// Derived from the flashv100 lineage (pinned be3722bc; ownership decision
// 2026-10-16: in-tree local implementation).

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "attention/qsa_page4_common.h"
#include "dispatch_utils.h"
#include "torch_utils.h"

#include <cub/block/block_radix_sort.cuh>

#include <climits>
#include <cstdint>

namespace vllm {
namespace qsa_page4 {

namespace {

constexpr int kGroupedSparseQueries = 8;
constexpr int kGroupedSparsePlannerThreads = 512;
constexpr int kGroupedSparseHashCapacity = 8192;
constexpr unsigned long long kGroupedSparseEmptyEntry = 0x00000000ffffffffULL;
constexpr int kGroupedSparseItemsPerThread =
    kGroupedSparseHashCapacity / kGroupedSparsePlannerThreads;
using GroupedSparseSort =
    cub::BlockRadixSort<unsigned long long, kGroupedSparsePlannerThreads,
                        kGroupedSparseItemsPerThread, unsigned long long>;
// Hash entries plus logical owners exactly fit Volta's 96 KiB opt-in limit.
// After loading both into registers, reuse this storage for sorting and scans.
constexpr size_t kGroupedSparsePlannerSharedMemory =
    kGroupedSparseHashCapacity *
    (sizeof(unsigned long long) + sizeof(uint32_t));
static_assert(sizeof(GroupedSparseSort::TempStorage) <=
              kGroupedSparsePlannerSharedMemory);

__device__ __forceinline__ void grouped_sparse_hash_insert(
    unsigned long long* __restrict__ hash_table,
    uint32_t* __restrict__ logical_owners, const int physical_microblock,
    const uint32_t token_mask, const int query, const int logical_token) {
  if (physical_microblock < 0 || token_mask == 0) {
    return;
  }
  int slot = (static_cast<uint32_t>(physical_microblock) * 2654435761u) &
             (kGroupedSparseHashCapacity - 1);
  const unsigned long long desired =
      (static_cast<unsigned long long>(token_mask) << 32) |
      static_cast<uint32_t>(physical_microblock);
#pragma unroll 1
  for (int probe = 0; probe < kGroupedSparseHashCapacity; ++probe) {
    const unsigned long long old =
        atomicCAS(hash_table + slot, kGroupedSparseEmptyEntry, desired);
    if (old == kGroupedSparseEmptyEntry ||
        static_cast<uint32_t>(old) ==
            static_cast<uint32_t>(physical_microblock)) {
      atomicOr(hash_table + slot, static_cast<unsigned long long>(token_mask)
                                      << 32);
      // Nonnegative int32 tokens use at most 29 bits after division by four.
      // The first contributing query owns shared pages, independent of request
      // slot IDs, physical allocation, insertion order, and hash collisions.
      const uint32_t owner = (static_cast<uint32_t>(query) << 29) |
                             (static_cast<uint32_t>(logical_token) >> 2);
      atomicMin(logical_owners + slot, owner);
      return;
    }
    slot = (slot + 1) & (kGroupedSparseHashCapacity - 1);
  }
}

__device__ __forceinline__ int grouped_sparse_physical_microblock(
    const int token, const int request_idx,
    const int* __restrict__ request_block_table,
    const int64_t request_block_table_stride, const int block_table_width,
    const int page_size, const int physical_page_stride,
    const int num_cache_blocks) {
  if (token < 0) {
    return -1;
  }
  const int logical_page = token / page_size;
  if (logical_page < 0 || logical_page >= block_table_width) {
    return -1;
  }
  const int page_offset = token - logical_page * page_size;
  const int physical_page =
      __ldg(request_block_table +
            static_cast<int64_t>(request_idx) * request_block_table_stride +
            logical_page);
  if (physical_page < 0 || physical_page >= num_cache_blocks) {
    return -1;
  }
  return physical_page * physical_page_stride + page_offset / 4;
}

__device__ __forceinline__ int grouped_sparse_active_m_tiles(
    const uint32_t token_mask) {
  int active_m_tiles = 0;
#pragma unroll
  for (int query = 0; query < kGroupedSparseQueries; ++query) {
    if ((token_mask & (0xFu << (query * 4))) != 0) {
      const int first_row = query * kGroupedVerifyHeads;
      const int last_row = first_row + kGroupedVerifyHeads - 1;
      active_m_tiles |= 1 << (first_row / 16);
      active_m_tiles |= 1 << (last_row / 16);
    }
  }
  return active_m_tiles;
}

__global__
__launch_bounds__(kGroupedSparsePlannerThreads, 1) void grouped_sparse_page4_plan_kernel(
    const int* __restrict__ logical_indices,
    const int* __restrict__ request_block_table,
    const int* __restrict__ token_to_req,
    const int64_t* __restrict__ query_positions,
    const int* __restrict__ sequence_lengths, int* __restrict__ output_blocks,
    uint32_t* __restrict__ output_masks, int* __restrict__ output_seq_lens,
    const int selection_width, const int64_t logical_indices_stride,
    const int64_t request_block_table_stride, const int num_requests,
    const int block_table_width, const int output_width, const int page_size,
    const int physical_page_stride, const int num_cache_blocks,
    // Per-query budget in 4-token candidate pages; 0 = no budget.
    // The indexer emits candidates in descending score order, so walking only
    // the first `page_budget` candidates is a score-ordered top-K selection:
    // it bounds the hash inserts (and therefore the union the fwd later
    // reads) instead of truncating the finished, position-ordered page list.
    const int page_budget) {
  const int group_idx = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ unsigned long long hash_table[];
  auto* logical_owners =
      reinterpret_cast<uint32_t*>(hash_table + kGroupedSparseHashCapacity);
  for (int slot = tid; slot < kGroupedSparseHashCapacity;
       slot += kGroupedSparsePlannerThreads) {
    hash_table[slot] = kGroupedSparseEmptyEntry;
    logical_owners[slot] = UINT_MAX;
  }
  __syncthreads();

  const int full_page4_count = selection_width / 4;
  const int page_limit =
      page_budget > 0 ? min(full_page4_count, page_budget) : full_page4_count;
  for (int selected_page = tid; selected_page < page_limit;
       selected_page += kGroupedSparsePlannerThreads) {
#pragma unroll
    for (int query = 0; query < kGroupedSparseQueries; ++query) {
      const int row = group_idx * kGroupedSparseQueries + query;
      const int request_idx = __ldg(token_to_req + row);
      if (request_idx < 0 || request_idx >= num_requests) {
        continue;
      }
      const int sequence_length = __ldg(sequence_lengths + request_idx);
      const int64_t query_visible_tokens = __ldg(query_positions + row) + 1;
      const int visible_tokens =
          query_visible_tokens <= 0
              ? 0
              : (query_visible_tokens < sequence_length
                     ? static_cast<int>(query_visible_tokens)
                     : max(sequence_length, 0));
      const int row_complete_page4_count =
          min(min(visible_tokens / 4, sequence_length / 4), full_page4_count);
      if (selected_page >= row_complete_page4_count) {
        continue;
      }
      const int* selected = logical_indices +
                            static_cast<int64_t>(row) * logical_indices_stride +
                            selected_page * 4;
      const int first_token = __ldg(selected);
      if (first_token < 0) {
        continue;
      }
      const bool full_page4 = __ldg(selected + 1) == first_token + 1 &&
                              __ldg(selected + 2) == first_token + 2 &&
                              __ldg(selected + 3) == first_token + 3 &&
                              (first_token & 3) == 0 &&
                              first_token + 3 < sequence_length;
      if (full_page4) {
        const int physical_microblock = grouped_sparse_physical_microblock(
            first_token, request_idx, request_block_table,
            request_block_table_stride, block_table_width, page_size,
            physical_page_stride, num_cache_blocks);
        grouped_sparse_hash_insert(hash_table, logical_owners,
                                   physical_microblock, 0xFu << (query * 4),
                                   query, first_token);
      } else {
#pragma unroll
        for (int token_offset = 0; token_offset < 4; ++token_offset) {
          const int token = __ldg(selected + token_offset);
          if (token >= 0 && token < sequence_length) {
            const int physical_microblock = grouped_sparse_physical_microblock(
                token, request_idx, request_block_table,
                request_block_table_stride, block_table_width, page_size,
                physical_page_stride, num_cache_blocks);
            grouped_sparse_hash_insert(
                hash_table, logical_owners, physical_microblock,
                1u << (query * 4 + (token & 3)), query, token);
          }
        }
      }
    }
  }
  if (tid < kGroupedSparseQueries) {
    const int query = tid;
    const int row = group_idx * kGroupedSparseQueries + query;
    const int request_idx = __ldg(token_to_req + row);
    if (request_idx >= 0 && request_idx < num_requests) {
      const int sequence_length = __ldg(sequence_lengths + request_idx);
      const int64_t query_visible_tokens = __ldg(query_positions + row) + 1;
      const int visible_tokens =
          query_visible_tokens <= 0
              ? 0
              : (query_visible_tokens < sequence_length
                     ? static_cast<int>(query_visible_tokens)
                     : max(sequence_length, 0));
      const int complete_page4_count =
          min(min(visible_tokens / 4, sequence_length / 4), full_page4_count);
      const int tail_count = visible_tokens & 3;
      const int tail_index = complete_page4_count * 4;
      const int selected_tail_token =
          tail_index < selection_width
              ? __ldg(logical_indices +
                      static_cast<int64_t>(row) * logical_indices_stride +
                      tail_index)
              : -1;
      const int expected_tail_token = (visible_tokens / 4) * 4;
      if (tail_count > 0 && selected_tail_token == expected_tail_token &&
          selected_tail_token < sequence_length) {
        const int physical_microblock = grouped_sparse_physical_microblock(
            selected_tail_token, request_idx, request_block_table,
            request_block_table_stride, block_table_width, page_size,
            physical_page_stride, num_cache_blocks);
        const uint32_t tail_mask = ((1u << tail_count) - 1) << (query * 4);
        grouped_sparse_hash_insert(hash_table, logical_owners,
                                   physical_microblock, tail_mask, query,
                                   selected_tail_token);
      }
    }
  }
  __syncthreads();

  // Keep the existing physical-page union and masks, but never let physical
  // hash slots determine the attention reduction order. Category is primary
  // to preserve active-tile packing; the logical owner orders each category.
  unsigned long long entries[kGroupedSparseItemsPerThread];
  unsigned long long sort_keys[kGroupedSparseItemsPerThread];
#pragma unroll
  for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
    const int slot = tid * kGroupedSparseItemsPerThread + item;
    const unsigned long long entry = hash_table[slot];
    entries[item] = entry;
    const int category =
        grouped_sparse_active_m_tiles(static_cast<uint32_t>(entry >> 32));
    sort_keys[item] = static_cast<uint32_t>(entry) == 0xffffffffu
                          ? ULLONG_MAX
                          : (static_cast<unsigned long long>(category) << 32) |
                                logical_owners[slot];
  }
  __syncthreads();
  auto& sort_storage =
      *reinterpret_cast<GroupedSparseSort::TempStorage*>(hash_table);
  // Three category bits, 32 owner bits, and one bit separating empty slots.
  GroupedSparseSort(sort_storage).Sort(sort_keys, entries, 0, 36);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
    hash_table[tid * kGroupedSparseItemsPerThread + item] = entries[item];
  }
  auto* category_counts = reinterpret_cast<int*>(logical_owners);
  int* category_offsets = category_counts + 8;
  int* category_cursors = category_offsets + 8;
  int* warp_category_prefix = category_cursors + 8;
  __syncthreads();

  if (tid < 8) {
    category_counts[tid] = 0;
    category_offsets[tid] = 0;
    category_cursors[tid] = 0;
  }
  __syncthreads();
  for (int slot = tid; slot < kGroupedSparseHashCapacity;
       slot += kGroupedSparsePlannerThreads) {
    const unsigned long long entry = hash_table[slot];
    if (static_cast<uint32_t>(entry) != 0xffffffffu) {
      const int category =
          grouped_sparse_active_m_tiles(static_cast<uint32_t>(entry >> 32));
      atomicAdd(category_counts + category, 1);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int padded_offset = 0;
#pragma unroll
    for (int category = 1; category < 8; ++category) {
      category_offsets[category] = padded_offset;
      padded_offset += (category_counts[category] + 7) & ~7;
    }
    category_offsets[0] = padded_offset;
  }
  __syncthreads();
  constexpr int kPlannerWarps = kGroupedSparsePlannerThreads / kWarpSize;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  for (int chunk_start = 0; chunk_start < kGroupedSparseHashCapacity;
       chunk_start += kGroupedSparsePlannerThreads) {
    const unsigned long long entry = hash_table[chunk_start + tid];
    const uint32_t physical_microblock = static_cast<uint32_t>(entry);
    int category = 0;
    if (physical_microblock != 0xffffffffu) {
      const uint32_t token_mask = static_cast<uint32_t>(entry >> 32);
      category = grouped_sparse_active_m_tiles(token_mask);
    }
    unsigned category_lanes = 0;
#pragma unroll
    for (int scan_category = 1; scan_category < 8; ++scan_category) {
      const unsigned lanes =
          __ballot_sync(0xffffffffu, category == scan_category);
      if (category == scan_category) {
        category_lanes = lanes;
      }
      if (lane == 0) {
        warp_category_prefix[warp * 8 + scan_category] = __popc(lanes);
      }
    }
    __syncthreads();
    if (tid > 0 && tid < 8) {
      int prefix = category_cursors[tid];
#pragma unroll
      for (int scan_warp = 0; scan_warp < kPlannerWarps; ++scan_warp) {
        const int count = warp_category_prefix[scan_warp * 8 + tid];
        warp_category_prefix[scan_warp * 8 + tid] = prefix;
        prefix += count;
      }
      category_cursors[tid] = prefix;
    }
    __syncthreads();
    if (category != 0) {
      const int category_rank = warp_category_prefix[warp * 8 + category] +
                                __popc(category_lanes & ((1u << lane) - 1));
      const int output_idx = category_offsets[category] + category_rank;
      if (output_idx < output_width) {
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] = static_cast<int>(physical_microblock);
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = static_cast<uint32_t>(entry >> 32);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (tid > 0 && tid < 8) {
    const int category = tid;
    const int padded_count = (category_counts[category] + 7) & ~7;
    for (int local_idx = category_counts[category]; local_idx < padded_count;
         ++local_idx) {
      const int output_idx = category_offsets[category] + local_idx;
      if (output_idx < output_width) {
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] = 0;
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = 0;
      }
    }
  }
  if (tid == 0) {
    output_seq_lens[group_idx] = min(category_offsets[0], output_width) * 4;
  }
}

}  // namespace

torch::stable::Tensor flash_attention_grouped_sparse_page4_plan(
    const torch::stable::Tensor& logical_indices,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& token_to_req,
    const torch::stable::Tensor& query_positions,
    const torch::stable::Tensor& sequence_lengths,
    torch::stable::Tensor& output_blocks, torch::stable::Tensor& output_masks,
    torch::stable::Tensor& output_seq_lens, const int page_size,
    const int physical_page_stride, const int num_cache_blocks,
    const int page_budget) {
  STD_TORCH_CHECK(
      logical_indices.is_cuda() && block_table.is_cuda() &&
          token_to_req.is_cuda() && query_positions.is_cuda() &&
          sequence_lengths.is_cuda() && output_blocks.is_cuda() &&
          output_masks.is_cuda() && output_seq_lens.is_cuda(),
      "grouped sparse page4 planner tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      logical_indices.scalar_type() ==
              torch::headeronly::ScalarType::Int &&
          block_table.scalar_type() == torch::headeronly::ScalarType::Int &&
          token_to_req.scalar_type() == torch::headeronly::ScalarType::Int &&
          query_positions.scalar_type() == torch::headeronly::ScalarType::Long &&
          sequence_lengths.scalar_type() == torch::headeronly::ScalarType::Int &&
          output_blocks.scalar_type() == torch::headeronly::ScalarType::Int &&
          output_masks.scalar_type() == torch::headeronly::ScalarType::UInt32 &&
          output_seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
      "grouped sparse page4 planner requires int32/uint32 metadata");
  STD_TORCH_CHECK(
      logical_indices.dim() == 2 && logical_indices.sizes()[0] > 0 &&
          logical_indices.sizes()[0] % kGroupedSparseQueries == 0 &&
          logical_indices.sizes()[1] == 2051,
      "grouped sparse page4 planner requires [8*N, 2051] indices");
  const int64_t num_groups =
      logical_indices.sizes()[0] / kGroupedSparseQueries;
  STD_TORCH_CHECK(
      block_table.dim() == 2 &&
          token_to_req.sizes() ==
              torch::headeronly::IntHeaderOnlyArrayRef(
                  {logical_indices.sizes()[0]}),
      "grouped sparse page4 planner request metadata is invalid");
  STD_TORCH_CHECK(
      query_positions.sizes() ==
          torch::headeronly::IntHeaderOnlyArrayRef(
              {logical_indices.sizes()[0]}) &&
          sequence_lengths.sizes() ==
              torch::headeronly::IntHeaderOnlyArrayRef(
                  {block_table.sizes()[0]}),
      "grouped sparse page4 planner visibility metadata is invalid");
  STD_TORCH_CHECK(
      output_blocks.dim() == 2 && output_blocks.sizes()[0] == num_groups &&
          output_blocks.sizes()[1] >= 4160 &&
          output_masks.sizes() == output_blocks.sizes() &&
          output_seq_lens.sizes() ==
              torch::headeronly::IntHeaderOnlyArrayRef({num_groups}),
      "grouped sparse page4 planner outputs must be [groups, >=4160]");
  STD_TORCH_CHECK(
      logical_indices.is_contiguous() && block_table.is_contiguous() &&
          token_to_req.is_contiguous() && query_positions.is_contiguous() &&
          sequence_lengths.is_contiguous() && output_blocks.is_contiguous() &&
          output_masks.is_contiguous() && output_seq_lens.is_contiguous(),
      "grouped sparse page4 planner metadata must be contiguous");
  STD_TORCH_CHECK(
      page_size > 0 && page_size % 4 == 0 && physical_page_stride > 0 &&
          num_cache_blocks > 0,
      "grouped sparse page4 planner requires page_size divisible by 4");
  STD_TORCH_CHECK(
      logical_indices.get_device_index() ==
              block_table.get_device_index() &&
          logical_indices.get_device_index() ==
              token_to_req.get_device_index() &&
          logical_indices.get_device_index() ==
              query_positions.get_device_index() &&
          logical_indices.get_device_index() ==
              sequence_lengths.get_device_index() &&
          logical_indices.get_device_index() ==
              output_blocks.get_device_index() &&
          logical_indices.get_device_index() == output_masks.get_device_index() &&
          logical_indices.get_device_index() ==
              output_seq_lens.get_device_index(),
      "grouped sparse page4 planner tensors must share one device");

  const torch::stable::accelerator::DeviceGuard device_guard(
      logical_indices.get_device_index());
  // Stable-ABI device-property check (in-tree precedent: torch_utils.h
  // device cache); the guard above makes this the tensor's device.
  const cudaDeviceProp* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 7 && properties->minor == 0,
                  "grouped sparse page4 planner supports SM70 only");
  const cudaStream_t stream =
      get_current_cuda_stream(logical_indices.get_device_index());
  const cudaError_t smem_status =
      cudaFuncSetAttribute(grouped_sparse_page4_plan_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kGroupedSparsePlannerSharedMemory);
  STD_TORCH_CHECK(
      smem_status == cudaSuccess,
      "Failed to set grouped sparse page4 planner shared memory: ",
      cudaGetErrorString(smem_status));
  grouped_sparse_page4_plan_kernel<<<
      static_cast<unsigned>(num_groups), kGroupedSparsePlannerThreads,
      kGroupedSparsePlannerSharedMemory, stream>>>(
      logical_indices.const_data_ptr<int>(),
      block_table.const_data_ptr<int>(), token_to_req.const_data_ptr<int>(),
      query_positions.const_data_ptr<int64_t>(),
      sequence_lengths.const_data_ptr<int>(),
      // Output tensors are written by the kernel, so use the mutable view.
      output_blocks.mutable_data_ptr<int>(),
      output_masks.mutable_data_ptr<uint32_t>(),
      output_seq_lens.mutable_data_ptr<int>(),
      static_cast<int>(logical_indices.sizes()[1]),
      logical_indices.strides()[0], block_table.strides()[0],
      static_cast<int>(block_table.sizes()[0]),
      static_cast<int>(block_table.sizes()[1]),
      static_cast<int>(output_blocks.sizes()[1]), page_size,
      physical_page_stride, num_cache_blocks, page_budget);
  const cudaError_t launch_error = cudaGetLastError();
  STD_TORCH_CHECK(launch_error == cudaSuccess,
                  "grouped sparse page4 planner kernel launch failed: ",
                  cudaGetErrorString(launch_error));
  return output_blocks;
}

int64_t flash_attention_grouped_sparse_page4_abi_version() {
  // Version 1 accepted FP16 K/V through the nine-argument forward binding.
  // Version 2 adds kv_cache_dtype and calibrated K/V scales.
  // Version 3 adds the planner's per-query page budget.
  return 3;
}

}  // namespace qsa_page4
}  // namespace vllm
