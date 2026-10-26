// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Torch op registration (stable ABI) for the in-tree QSA page4 kernels.
// The in-service surface (vllm/models/qwen4_exp/nvidia/ops/qsa.py) consumes
// these ops by name through torch.ops._C; the plan and abi ops are the
// prefill-side entries (planner unit), the fwd/XQA entries join as their
// port units land (localization P0, 10-16).

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <cstdint>
#include <optional>
#include <string>

namespace vllm {
namespace qsa_page4 {

// "float" schema args bind to C++ double (stable-ABI convention); the
// fp32 kernel path casts at the launch site.
torch::stable::Tensor flash_attention_grouped_sparse_page4(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& token_masks,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& lse,
    const double softmax_scale, const std::string& kv_cache_dtype,
    const double k_scale, const double v_scale);

torch::stable::Tensor flash_attention_grouped_sparse_page4_plan(
    const torch::stable::Tensor& logical_indices,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& token_to_req,
    const torch::stable::Tensor& query_positions,
    const torch::stable::Tensor& sequence_lengths,
    torch::stable::Tensor& output_blocks, torch::stable::Tensor& output_masks,
    torch::stable::Tensor& output_seq_lens, const int page_size,
    const int physical_page_stride, const int num_cache_blocks,
    const int page_budget);

int64_t flash_attention_grouped_sparse_page4_abi_version();

// 20-arg XQA entry (XQA unit); the 19-arg in-service ABI drops the
// optional anchor_lens (nullopt) and the "float" schema args bind to
// double, so the boxed impl is the 19-arg forwarder below.
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
    const int64_t anchored_window);

// 18-arg XQA entry (XQA unit); in-tree binding in
// qsa_page4_xqa_binding.cu. The torch op's schema "float" args bind to
// double, so the scale params are double here (the binding casts to float).
torch::stable::Tensor flash_attention_decode_paged_xqa(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    std::optional<torch::stable::Tensor> out_,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const double softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const double k_scale, const double v_scale, const int window_size_left,
    const int window_size_right, const int batch_context_max_seq_len);

torch::stable::Tensor qsa_decode_paged_xqa_fwd(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache, torch::stable::Tensor& out,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const double softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const double k_scale, const double v_scale, const int window_size_left,
    const int window_size_right, const int64_t anchored_window) {
  return flash_attention_decode_paged_xqa(
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, active_num_partitions, softmax_scale, partition_size,
      launch_num_partitions, kv_cache_dtype, k_scale, v_scale,
      window_size_left, window_size_right,
      static_cast<int>(anchored_window));
}

}  // namespace qsa_page4
}  // namespace vllm

using namespace vllm::qsa_page4;

STABLE_TORCH_LIBRARY_FRAGMENT(_C, qsa_page4_ops) {
  qsa_page4_ops.def(
      "grouped_sparse_page4_plan_fwd(Tensor logical_indices, "
      "Tensor block_table, Tensor token_to_req, Tensor query_positions, "
      "Tensor sequence_lengths, Tensor output_blocks, Tensor output_masks, "
      "Tensor output_seq_lens, int page_size, int physical_page_stride, "
      "int num_cache_blocks, int page_budget) -> Tensor");
  qsa_page4_ops.def("grouped_sparse_page4_abi_version() -> int");
  qsa_page4_ops.def(
      "grouped_sparse_page4_fwd(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor out, Tensor block_table, Tensor token_masks, Tensor seq_lens, "
      "Tensor lse, float softmax_scale, str kv_cache_dtype, float k_scale, "
      "float v_scale) -> Tensor");
  qsa_page4_ops.def(
      "decode_paged_xqa_fwd(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor out, Tensor block_table, Tensor seq_lens, Tensor tmp_out, "
      "Tensor max_logits, Tensor exp_sums, "
      "Tensor active_num_partitions, float softmax_scale, "
      "int partition_size, int launch_num_partitions, str kv_cache_dtype, "
      "float k_scale, float v_scale, int window_size_left, "
      "int window_size_right, int anchored_window) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, qsa_page4_ops) {
  qsa_page4_ops.impl("grouped_sparse_page4_plan_fwd",
                     TORCH_BOX(&::flash_attention_grouped_sparse_page4_plan));
  qsa_page4_ops.impl("grouped_sparse_page4_fwd",
                     TORCH_BOX(&::flash_attention_grouped_sparse_page4));
  qsa_page4_ops.impl("decode_paged_xqa_fwd",
                     TORCH_BOX(&::qsa_decode_paged_xqa_fwd));
}

// Zero-tensor-argument ops must dispatch on CompositeExplicitAutograd
// (in-tree precedent: sm70_hier_ar.cu), not CUDA.
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, qsa_page4_ops) {
  qsa_page4_ops.impl("grouped_sparse_page4_abi_version",
                     TORCH_BOX(&::flash_attention_grouped_sparse_page4_abi_version));
}
