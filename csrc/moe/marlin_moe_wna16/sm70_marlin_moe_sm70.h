// SM70 (V100/Volta) NVFP4 MoE Marlin GEMM entry point.
// Implemented in sm70_marlin_nvfp4_gemm.cu; called from ops.cu when the
// device is SM70 and the quantization is NVFP4 (group size 16).
#pragma once

#include <torch/csrc/stable/tensor.h>

namespace marlin_moe_wna16 {

torch::stable::Tensor sm70_marlin_nvfp4_gemm(
    torch::stable::Tensor& a, torch::stable::Tensor& c,
    torch::stable::Tensor& b_q_weight, torch::stable::Tensor& b_scales,
    torch::stable::Tensor& global_scale,
    torch::stable::Tensor& sorted_token_ids, torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& num_tokens_past_padded,
    torch::stable::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size);

}  // namespace marlin_moe_wna16
