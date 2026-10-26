// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// In-tree binding for the in-service XQA decode entry
// `flash_attention_decode_paged_xqa` (flashv100 lineage, source
// kernel/flash_decode_paged.cu L5162-6144). The sibling
// `flash_attention_decode_paged` was already ported to qsa_page4_xqa.cu; this
// file adds the missing XQA dispatch so the local torch op
// (qsa_decode_paged_xqa_fwd) calls the SAME C++ function the in-service .so
// calls.
//
// The XQA kernels and launchers already live in qsa_page4_xqa.cu. Those
// launchers are (non-inline) function templates that must be instantiated in
// the calling translation unit, so this binding file includes that unit to
// reuse them (single-TU reuse; see the note at the bottom about the final
// link). It then adds only the pieces the source expected from the flashv100
// policy header (minimal stand-ins with identical env-var names/defaults)
// plus the verbatim-transcribed `flash_attention_decode_paged_xqa` with a
// stable-ABI conversion:
//   * at::Tensor -> torch::stable::Tensor;  .size(i) -> .sizes()[i];
//     .stride(i) -> .strides()[i];  .sizes()== -> .sizes().equals();
//   * .data_ptr<T>() -> .const_data_ptr<T>() (reads);  for the fp16 output the
//     raw .mutable_data_ptr() + reinterpret_cast<__half*> (no stable
//     mutable_data_ptr<__half> specialization);
//   * TORCH_CHECK -> STD_TORCH_CHECK;  dtype() -> scalar_type() ==
//     torch::headeronly::ScalarType::*;  at::Half/at::Byte -> headeronly
//     ScalarType::Half/Byte;
//   * c10::cuda::CUDAGuard -> torch::stable::accelerator::DeviceGuard;
//     at::cuda::getCurrentCUDAStream().stream() -> get_current_cuda_stream();
//     at::cuda::getDeviceProperties(d)->multiProcessorCount ->
//     get_device_prop()->multiProcessorCount;  C10_CUDA_KERNEL_LAUNCH_CHECK ->
//     cuda_launch_check();
//   * schema "float" args arrive as double (stable-ABI convention) -> cast to
//     float once, at the top of the function;
//   * out_ keeps the source's has_value()/empty_like fallback (matches the
//     existing flash_attention_decode_paged port).
#include "attention/qsa_page4_xqa.cu"

namespace vllm {
namespace qsa_page4 {

// ---- Policy stand-ins (flash_v100_policy.h is deliberately not ported) ----
// Each mirrors the direct-call getenv semantics of the corresponding Field in
// the reference header (identical env var name, identical default, identical
// parse). Defaults reflect the reference: "enabled" fields default on unless
// explicitly "0"; "trace/dense/split/require" fields default off unless "1";
// numeric fields parse with std::atoi (clamped/validated exactly as the
// reference).
inline bool xqa_padded_smem_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_PADDED_SMEM");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_g6_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_e4m3_batch_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_BATCH_XQA");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e4m3_batch_optimized_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_BATCH_XQA_OPTIMIZED");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e4m3_page800_fastpath_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e4m3_g6_p64_p256_auto_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e4m3_g6_p64_p256_auto_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE");
  return value != nullptr && value[0] == '1';
}

inline int xqa_e4m3_g6_p256_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P256_BEGIN");
  return value == nullptr ? 12288 : std::max(1, std::atoi(value));
}

inline int xqa_e4m3_g6_dual_cta_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_DUAL_CTA_BEGIN");
  return value == nullptr ? 32768 : std::max(1, std::atoi(value));
}

inline bool xqa_e4m3_g6_wave_partitions_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e4m3_g6_merged_wave_launch_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH");
  return value == nullptr || value[0] != '0';
}

inline int xqa_e4m3_g6_p512_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P512_BEGIN");
  return value == nullptr ? 49152 : std::max(1, std::atoi(value));
}

inline int xqa_e4m3_g6_p896_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P896_BEGIN");
  return value == nullptr ? 98304 : std::max(1, std::atoi(value));
}

inline int xqa_e4m3_g6_p1664_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E4M3_G6_P1664_BEGIN");
  return value == nullptr ? 196608 : std::max(1, std::atoi(value));
}

inline bool decode_partition_size_overridden() {
  return std::getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE") != nullptr;
}

inline bool xqa_g6_qk_pipeline_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE");
  return value == nullptr || value[0] != '0';
}

inline int xqa_g6_qk_pipeline_warps() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_WARPS");
  return value != nullptr && std::atoi(value) == 6 ? 6 : 8;
}

inline bool xqa_g6_qk_pipeline_trace_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_QK_PIPELINE_TRACE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_g6_p1024_sawtooth_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_TRACE");
  return value != nullptr && value[0] == '1';
}

inline int xqa_g6_p1024_sawtooth_p1024_mid_seq_len() {
  const char* value = std::getenv(
      "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_MID_SEQ_LEN");
  return value == nullptr ? 111104 : std::max(1, std::atoi(value));
}

inline int xqa_g6_p1024_sawtooth_p256_long_seq_len() {
  const char* value = std::getenv(
      "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P256_LONG_SEQ_LEN");
  return value == nullptr ? 147841 : std::max(1, std::atoi(value));
}

inline int xqa_g6_p1024_sawtooth_p1024_final_seq_len() {
  const char* value = std::getenv(
      "VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH_P1024_FINAL_SEQ_LEN");
  return value == nullptr ? 258176 : std::max(1, std::atoi(value));
}

inline bool xqa_split_reduce_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_SPLIT_REDUCE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_batch_context_routing_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE");
  return value != nullptr && value[0] == '1';
}

inline int xqa_block16_layout_mode() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT");
  if (value == nullptr) {
    return 0;
  }
  const int mode = std::atoi(value);
  return mode == 1 || mode == 2 ? mode : 0;
}

inline bool xqa_block16_layout_required() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_block16_layout_trace_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_block784_index_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK784_INDEX");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_block784_index_trace_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_aligned_padded_smem_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_aligned_padded_smem_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE");
  return value != nullptr && value[0] == '1';
}

inline int xqa_split_reduce_dim_tile() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE");
  if (value == nullptr) {
    return 8;
  }
  const int dim_tile = std::atoi(value);
  return dim_tile == 8 || dim_tile == 16 || dim_tile == 32 ? dim_tile : 8;
}

inline bool xqa_mtp5_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA");
  return value == nullptr || value[0] == '1';
}

inline bool xqa_g6_dual_cta_dense_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_g6_p1024_auto_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_AUTO");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_g6_p1024_auto_trace_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_AUTO_TRACE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_g6_p1024_sawtooth_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e5m2_g6_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e5m2_g6_split_reduce_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_G6_SPLIT_REDUCE");
  return value == nullptr || value[0] != '0';
}

inline int xqa_e5m2_p1024_begin() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_P1024_BEGIN");
  return value == nullptr ? 61633 : std::max(1, std::atoi(value));
}

inline int xqa_e5m2_scalar_xqa_seq_len() {
  const char* value = std::getenv("VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN");
  return value == nullptr ? 16384 : std::max(1, std::atoi(value));
}

inline bool xqa_e5m2_g6_dual_cta_trace_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E5M2_G6_DUAL_CTA_TRACE");
  return value != nullptr && value[0] == '1';
}

inline bool xqa_e5m2_partition_page_ids_enabled() {
  const char* value =
      std::getenv("VLLM_FLASH_V100_XQA_E5M2_PARTITION_PAGE_IDS");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e5m2_pair_load_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_PAIR_LOAD");
  return value == nullptr || value[0] != '0';
}

inline bool xqa_e5m2_batch_wide_load_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD");
  return value == nullptr || value[0] != '0';
}

// ---- Batch-context route (transcribed from the reference kernel file) ----
enum class XQABatchContextRoute : int {
  kDisabled = -1,
  kBaseline = 0,
  kDualCta = 1,
  kDualCtaSplit = 2,
};

inline bool xqa_batch_context_routing_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING");
  return value == nullptr || value[0] != '0';
}

XQABatchContextRoute select_xqa_batch_context_route(const int batch_size,
                                                    const int max_seq_len,
                                                    const int partition_size) {
  if (!xqa_batch_context_routing_enabled() || batch_size < 4 ||
      max_seq_len <= 0) {
    return XQABatchContextRoute::kDisabled;
  }
  if (batch_size <= 4) {
    if (max_seq_len <= 8191) {
      return XQABatchContextRoute::kBaseline;
    }
    if (max_seq_len <= 12287) {
      return XQABatchContextRoute::kDualCta;
    }
    return partition_size == 1024 ? XQABatchContextRoute::kBaseline
                                  : XQABatchContextRoute::kDualCtaSplit;
  }
  if (batch_size <= 8) {
    if (max_seq_len <= 4095) {
      return XQABatchContextRoute::kBaseline;
    }
    return max_seq_len <= 12287 ? XQABatchContextRoute::kDualCta
                                : XQABatchContextRoute::kDualCtaSplit;
  }
  if (batch_size <= 12) {
    if (max_seq_len <= 2047) {
      return XQABatchContextRoute::kBaseline;
    }
    return max_seq_len <= 16000 ? XQABatchContextRoute::kDualCta
                                : XQABatchContextRoute::kDualCtaSplit;
  }
  return max_seq_len <= 12287 ? XQABatchContextRoute::kBaseline
                              : XQABatchContextRoute::kDualCta;
}

// Trace functions are diagnostics only; keep them as no-ops (the reference
// emits TORCH_WARN; the in-tree stable-ABI build has no such macro and the
// *_trace_enabled() stand-ins default to disabled).
inline void trace_xqa_batch_context_route(const int batch_size,
                                          const int max_seq_len,
                                          const int partition_size,
                                          const int block_size,
                                          const XQABatchContextRoute route) {}

inline void trace_xqa_e4m3_page800_fastpath(const int batch_size,
                                           const int partition_size,
                                           const bool active) {}

// Stand-in for flash_v100::policy::observation(slot): per-slot diagnostic
// counters. The trace blocks that read these are gated behind
// *_trace_enabled() stand-ins (default off), so they are never written in
// practice.
inline std::atomic<unsigned long long>& xqa_policy_observation(
    const size_t slot) {
  static std::array<std::atomic<unsigned long long>, 16> slots{};
  return slots[slot];
}

// No stable-ABI TORCH_WARN exists; the diagnostics become no-ops. Defined
// around the function body so the verbatim trace blocks still parse.
#ifdef TORCH_WARN
#undef TORCH_WARN
#endif
#define TORCH_WARN(...) static_cast<void>(0)

torch::stable::Tensor flash_attention_decode_paged_xqa(
    const torch::stable::Tensor& q, const torch::stable::Tensor& k_cache,
    const torch::stable::Tensor& v_cache,
    std::optional<torch::stable::Tensor> out_,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& exp_sums,
    const torch::stable::Tensor& active_num_partitions,
    const double _softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const double _k_scale, const double _v_scale, const int window_size_left,
    const int window_size_right, const int batch_context_max_seq_len) {
  // Schema "float" args arrive as double under the stable ABI; the fp32
  // kernel launchers take float.
  const float softmax_scale = static_cast<float>(_softmax_scale);
  const float k_scale = static_cast<float>(_k_scale);
  const float v_scale = static_cast<float>(_v_scale);
  STD_TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  STD_TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
                  "k_cache and v_cache must be on CUDA");
  STD_TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
                  "block_table and seq_lens must be on CUDA");
  STD_TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
                  "decode workspaces must be on CUDA");
  STD_TORCH_CHECK(active_num_partitions.is_cuda(),
                  "active_num_partitions must be on CUDA");
  STD_TORCH_CHECK(
      q.scalar_type() == torch::headeronly::ScalarType::Half, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  STD_TORCH_CHECK(kv_dtype_code == KV_CACHE_DTYPE_FP16 ||
                      kv_dtype_code == KV_CACHE_DTYPE_FP8_E4M3 ||
                      kv_dtype_code == KV_CACHE_DTYPE_FP8_E5M2 ||
                      kv_dtype_code == KV_CACHE_DTYPE_NVFP4,
                  "XQA decode supports fp16, fp8_e4m3, fp8_e5m2 and nvfp4 KV "
                  "cache only");
  if (kv_dtype_code == KV_CACHE_DTYPE_FP16) {
    STD_TORCH_CHECK(
        k_cache.scalar_type() == torch::headeronly::ScalarType::Half &&
            v_cache.scalar_type() == torch::headeronly::ScalarType::Half,
        "fp16 XQA requires fp16 K/V tensors");
  } else {
    STD_TORCH_CHECK(
        k_cache.scalar_type() == torch::headeronly::ScalarType::Byte &&
            v_cache.scalar_type() == torch::headeronly::ScalarType::Byte,
        "FP8 XQA requires uint8 K/V tensors");
  }
  STD_TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                  "XQA K/V scales must be positive");
  STD_TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int,
      "block_table must be int32");
  STD_TORCH_CHECK(seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
                  "seq_lens must be int32");
  STD_TORCH_CHECK(
      active_num_partitions.scalar_type() ==
          torch::headeronly::ScalarType::Int,
      "active_num_partitions must be int32");
  STD_TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                  "XQA decode does not support sliding-window attention");
  STD_TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  STD_TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                  "KV cache must have shape [num_blocks, block_size, H_kv, D]");
  STD_TORCH_CHECK(block_table.dim() == 2,
                  "block_table must have shape [B, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  STD_TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  STD_TORCH_CHECK(q.strides()[q.dim() - 1] == 1,
                  "q last dim must be contiguous");
  STD_TORCH_CHECK(
      k_cache.strides()[k_cache.dim() - 1] == 1 &&
          v_cache.strides()[v_cache.dim() - 1] == 1,
      "KV cache last dim must be contiguous");
  STD_TORCH_CHECK(q.sizes()[0] <= block_table.sizes()[0],
                  "block_table batch size must cover q batch size");
  STD_TORCH_CHECK(q.sizes()[0] <= seq_lens.sizes()[0],
                  "seq_lens batch size must cover q batch size");
  STD_TORCH_CHECK(k_cache.sizes().equals(v_cache.sizes()),
                  "K/V cache shape mismatch");
  if (kv_dtype_code == KV_CACHE_DTYPE_NVFP4) {
    // NVFP4 pool side view: [P, N, 1, 144] uint8 (kNvfp4* constants in
    // qsa_page4_kv_utils.cu); per (token, side) cell = [data 128 | scale 16],
    // token stride 288 (K+V cells), block stride = allocator page (padding
    // allowed, so no divisibility check). The last dim carries data + scale,
    // so the head_dim equality does not apply.
    STD_TORCH_CHECK(
        k_cache.sizes()[1] % 4 == 0 && k_cache.sizes()[2] == 1 &&
            k_cache.sizes()[3] == kNvfp4CellBytes &&
            k_cache.strides()[0] >= k_cache.sizes()[1] * kNvfp4TokenCellBytes &&
            k_cache.strides()[1] == kNvfp4TokenCellBytes &&
            k_cache.strides()[2] == kNvfp4CellBytes,
        "NVFP4 XQA requires the [pool_blocks, N, 1, 144] cell pool view");
  } else {
    STD_TORCH_CHECK(k_cache.sizes()[3] == q.sizes()[2],
                    "KV head_dim mismatch");
  }
  STD_TORCH_CHECK(q.sizes()[2] == 256, "XQA decode supports head_dim=256 only");
  const int num_heads_q = q.sizes()[1];
  const int num_heads_kv = k_cache.sizes()[2];
  STD_TORCH_CHECK(num_heads_kv > 0 && num_heads_q % num_heads_kv == 0,
                  "num_heads_q must be divisible by num_heads_kv");
  const int q_per_kv = num_heads_q / num_heads_kv;
  STD_TORCH_CHECK(q_per_kv == 4 || q_per_kv == 6 || q_per_kv == 8,
                  "XQA decode supports q_per_kv in {4, 6, 8}, got ", q_per_kv);
  STD_TORCH_CHECK(partition_size == 16 || partition_size == 32 ||
                      partition_size == 64 || partition_size == 128 ||
                      partition_size == 256 || partition_size == 512 ||
                      partition_size == 1024 ||
                      ((partition_size == 896 || partition_size == 1664) &&
                       kv_dtype_code == KV_CACHE_DTYPE_FP8_E4M3),
                  "Unsupported XQA decode partition_size: ", partition_size);
  STD_TORCH_CHECK(launch_num_partitions > 0,
                  "launch_num_partitions must be positive");
  const auto expected_partial_dtype =
      kv_dtype_code == KV_CACHE_DTYPE_FP8_E4M3
          ? torch::headeronly::ScalarType::Float
          : torch::headeronly::ScalarType::Half;
  STD_TORCH_CHECK(tmp_out.scalar_type() == expected_partial_dtype,
                  "XQA decode tmp_out must be fp32 for E4M3 KV and fp16 "
                  "otherwise");
  STD_TORCH_CHECK(
      tmp_out.sizes()[0] >= q.sizes()[0] &&
          tmp_out.sizes()[1] >= q.sizes()[1] &&
          tmp_out.sizes()[2] >= launch_num_partitions &&
          tmp_out.sizes()[3] == q.sizes()[2],
      "tmp_out shape does not cover XQA launch");
  STD_TORCH_CHECK(
      max_logits.sizes()[0] >= q.sizes()[0] &&
          max_logits.sizes()[1] >= q.sizes()[1] &&
          max_logits.sizes()[2] >= launch_num_partitions,
      "max_logits shape does not cover XQA launch");
  STD_TORCH_CHECK(
      exp_sums.sizes()[0] >= q.sizes()[0] &&
          exp_sums.sizes()[1] >= q.sizes()[1] &&
          exp_sums.sizes()[2] >= launch_num_partitions,
      "exp_sums shape does not cover XQA launch");

  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
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
  if (kv_dtype_code == KV_CACHE_DTYPE_FP8_E4M3) {
    const bool e4m3_batch_allowed = q.sizes()[0] > 1 && xqa_e4m3_batch_enabled();
    const bool e4m3_aligned_page =
        k_cache.sizes()[1] >= 256 && k_cache.sizes()[1] % 16 == 0;
    const bool use_large_partition =
        q.sizes()[0] == 1 &&
        (partition_size == 512 || partition_size == 896 ||
         partition_size == 1024 || partition_size == 1664) &&
        e4m3_aligned_page;
    STD_TORCH_CHECK((q.sizes()[0] == 1 || e4m3_batch_allowed) && q_per_kv == 6 &&
                        (partition_size == 64 || partition_size == 128 ||
                         partition_size == 256 || use_large_partition),
                    "E4M3 XQA supports B=1, or B>=2 when "
                    "VLLM_FLASH_V100_E4M3_BATCH_XQA=1; q_per_kv=6 and D=256 are "
                    "required. A 16-aligned page with at least 256 tokens and "
                    "additionally supports B1 partition sizes 512, 896, "
                    "1024, and 1664");
    const bool e4m3_batch_optimized =
        q.sizes()[0] > 1 && xqa_e4m3_batch_optimized_enabled() &&
        k_cache.sizes()[1] >= 256 && k_cache.sizes()[1] % 16 == 0 &&
        k_cache.scalar_type() == torch::headeronly::ScalarType::Byte;
    const XQABatchContextRoute e4m3_batch_route =
        e4m3_batch_optimized && batch_context_max_seq_len > 0 &&
                q.sizes()[0] >= 4
            ? select_xqa_batch_context_route(
                  q.sizes()[0], batch_context_max_seq_len, partition_size)
            : XQABatchContextRoute::kDisabled;
    const bool e4m3_dual_cta =
        e4m3_batch_route == XQABatchContextRoute::kDualCta ||
        e4m3_batch_route == XQABatchContextRoute::kDualCtaSplit;
    const bool e4m3_split_reduce =
        e4m3_batch_route == XQABatchContextRoute::kDualCtaSplit;
    constexpr int64_t kPage800HeadDim256Elements = 800 * 256;
    const bool e4m3_page800_contiguous =
        xqa_e4m3_page800_fastpath_enabled() && k_cache.sizes()[1] == 800 &&
        k_cache.sizes()[2] == 1 && k_cache.sizes()[3] == 256 &&
        k_cache.strides()[0] == 2 * kPage800HeadDim256Elements &&
        k_cache.strides()[1] == 256 && k_cache.strides()[2] == 256 &&
        k_cache.strides()[3] == 1 &&
        v_cache.strides()[0] == 2 * kPage800HeadDim256Elements &&
        v_cache.strides()[1] == 256 && v_cache.strides()[2] == 256 &&
        v_cache.strides()[3] == 1;
    trace_xqa_batch_context_route(q.sizes()[0], batch_context_max_seq_len,
                                  partition_size, k_cache.sizes()[1],
                                  e4m3_batch_route);
    trace_xqa_e4m3_page800_fastpath(q.sizes()[0], partition_size,
                                    e4m3_dual_cta && e4m3_page800_contiguous);
    const int split_reduce_dim_tile = xqa_split_reduce_dim_tile();

#define LAUNCH_E4M3_DUAL_CTA(PARTITION, PAGE_SIZE, CONTIGUOUS)              \
  launch_flash_attention_decode_paged_xqa_tc_256_wide<                      \
      PARTITION, 6, true, kXQATCG6DualCtaThreads, 2, PAGE_SIZE, CONTIGUOUS, \
      false, kXQARouteAllSeqLens, false, true, true,                        \
      KV_CACHE_DTYPE_FP8_E4M3>(                                             \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits, \
      exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,     \
      launch_num_partitions, e4m3_split_reduce, split_reduce_dim_tile, stream)

#define LAUNCH_E4M3_BATCH_XQA(PARTITION)                                       \
  do {                                                                         \
    if (e4m3_dual_cta) {                                                       \
      if (e4m3_page800_contiguous) {                                           \
        LAUNCH_E4M3_DUAL_CTA(PARTITION, 800, true);                            \
      } else {                                                                 \
        LAUNCH_E4M3_DUAL_CTA(PARTITION, 0, false);                             \
      }                                                                        \
    } else if (e4m3_batch_optimized) {                                         \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<                     \
          PARTITION, 6, true, kXQATC256WideThreads, 1, 0, false, false,        \
          kXQARouteAllSeqLens, false, true, true,                              \
          KV_CACHE_DTYPE_FP8_E4M3>(                                            \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, split_reduce_dim_tile,        \
          stream);                                                             \
    } else {                                                                   \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<                     \
          PARTITION, 6, true, kXQATC256WideThreads, 1, 0, false, false,        \
          kXQARouteAllSeqLens, false, false, false,                            \
          KV_CACHE_DTYPE_FP8_E4M3>(                                            \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    }                                                                          \
  } while (0)

    const bool use_p64_p256_auto = q.sizes()[0] == 1 && partition_size == 64 &&
                                   e4m3_aligned_page &&
                                   !decode_partition_size_overridden() &&
                                   xqa_e4m3_g6_p64_p256_auto_enabled();
    const int64_t page_head_dim_elements = k_cache.sizes()[1] * 256;
    const bool fixed_interleaved_hkv1_layout =
        (k_cache.sizes()[1] == 800 || k_cache.sizes()[1] == 1568) &&
        k_cache.sizes()[3] == 256 &&
        k_cache.strides()[0] == 2 * page_head_dim_elements &&
        k_cache.strides()[1] == 256 && k_cache.strides()[2] == 256 &&
        k_cache.strides()[3] == 1 &&
        v_cache.strides()[0] == 2 * page_head_dim_elements &&
        v_cache.strides()[1] == 256 && v_cache.strides()[2] == 256 &&
        v_cache.strides()[3] == 1;
    if (use_p64_p256_auto) {
      const int p256_begin = xqa_e4m3_g6_p256_begin();
      const int p512_begin = xqa_e4m3_g6_p512_begin();
      const bool use_wave_partitions =
          xqa_e4m3_g6_wave_partitions_enabled() &&
          batch_context_max_seq_len >= p512_begin;
      if (use_wave_partitions) {
        const int p896_begin = xqa_e4m3_g6_p896_begin();
        const int p1664_begin = xqa_e4m3_g6_p1664_begin();
        STD_TORCH_CHECK(
            p256_begin < p512_begin && p512_begin < p896_begin &&
                p896_begin < p1664_begin,
            "Invalid E4M3 wave-partition thresholds: ", p256_begin, ", ",
            p512_begin, ", ", p896_begin, ", ", p1664_begin);

        const int p64_launch_num_partitions =
            std::max(1, std::min(launch_num_partitions, (p256_begin + 62) / 64));
        const int p256_launch_num_partitions =
            (launch_num_partitions + 3) / 4;
        const int p512_launch_num_partitions =
            (launch_num_partitions + 7) / 8;
        const int p896_launch_num_partitions =
            (launch_num_partitions + 13) / 14;
        const int p1664_launch_num_partitions =
            (launch_num_partitions + 25) / 26;
        const int p256_mid_launch_num_partitions = std::max(
            1, std::min(p256_launch_num_partitions, (p512_begin + 254) / 256));
        const int short_launch_num_partitions =
            std::max(p64_launch_num_partitions, p256_mid_launch_num_partitions);
        const int p512_route_launch_num_partitions =
            std::max(1,
                     std::min(p512_launch_num_partitions,
                              (p896_begin + 510) / 512));
        const int p896_route_launch_num_partitions =
            std::max(1,
                     std::min(p896_launch_num_partitions,
                              (p1664_begin + 894) / 896));
        const int wave_long_launch_num_partitions = std::max(
            {p512_route_launch_num_partitions, p896_route_launch_num_partitions,
             p1664_launch_num_partitions});
        const bool use_merged_wave_launch =
            xqa_e4m3_g6_merged_wave_launch_enabled();

        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            0, 6, true, kXQATC256WideThreads, 1, 0, false, false,
            kXQARouteShortSeqLens, false, false, false,
            KV_CACHE_DTYPE_FP8_E4M3>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, short_launch_num_partitions, false, 8, stream, p512_begin,
            p256_begin, 0, false);
        if (use_merged_wave_launch) {
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<
              -1, kXQARouteWaveLongSeqLens, true>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, wave_long_launch_num_partitions, stream,
              fixed_interleaved_hkv1_layout, p512_begin, p896_begin,
              p1664_begin, false);
        } else {
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<
              512, kXQARouteRangeSeqLens>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, p512_route_launch_num_partitions, stream, false,
              p512_begin, p896_begin, 0, false);
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<
              896, kXQARouteRangeSeqLens>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, p896_route_launch_num_partitions, stream, false,
              p896_begin, p1664_begin, 0, false);
          launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<
              1664, kXQARouteLongSeqLens>(
              q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
              max_logits, exp_sums, active_num_partitions, softmax_scale,
              k_scale, v_scale, p1664_launch_num_partitions, stream, false,
              p1664_begin, 0, 0, false);
        }

        const dim3 reduce_grid(q.sizes()[0], q.sizes()[1], 1);
        const dim3 reduce_block(kThreadsPerBlock);
#define LAUNCH_E4M3_WAVE_REDUCER(PARTITION_SIZE, SEQ_ROUTE, MAX_PARTITIONS,  \
                                 PARTITION_BEGIN, ROUTE_BEGIN, ROUTE_END,    \
                                 ROUTE_FINAL)                                \
        do {                                                                 \
          const size_t reduce_shared_mem =                                   \
              static_cast<size_t>(2 * (MAX_PARTITIONS)) * sizeof(float);     \
          flash_attention_decode_reduce_kernel<                              \
              256, PARTITION_SIZE, SEQ_ROUTE, float>                         \
              <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(     \
                  tmp_out.const_data_ptr<float>(),                           \
                  max_logits.const_data_ptr<float>(),                        \
                  exp_sums.const_data_ptr<float>(),                          \
                  seq_lens.const_data_ptr<int>(),                            \
                  active_num_partitions.const_data_ptr<int>(),               \
                  reinterpret_cast<__half*>(out.mutable_data_ptr()),         \
                  q.sizes()[0], MAX_PARTITIONS, q.sizes()[1],                \
                  tmp_out.strides()[0], tmp_out.strides()[1],                \
                  tmp_out.strides()[2], max_logits.strides()[0],             \
                  max_logits.strides()[1], out.strides()[0], out.strides()[1], \
                  PARTITION_BEGIN, ROUTE_BEGIN, ROUTE_END, ROUTE_FINAL);      \
        } while (0)
        LAUNCH_E4M3_WAVE_REDUCER(0, kXQARouteShortSeqLens,
                                 short_launch_num_partitions, p256_begin,
                                 p512_begin, 0, 0);
        if (use_merged_wave_launch) {
          LAUNCH_E4M3_WAVE_REDUCER(-1, kXQARouteWaveLongSeqLens,
                                   wave_long_launch_num_partitions, 0,
                                   p512_begin, p896_begin, p1664_begin);
        } else {
          LAUNCH_E4M3_WAVE_REDUCER(512, kXQARouteRangeSeqLens,
                                   p512_route_launch_num_partitions, 0,
                                   p512_begin, p896_begin, 0);
          LAUNCH_E4M3_WAVE_REDUCER(896, kXQARouteRangeSeqLens,
                                   p896_route_launch_num_partitions, 0,
                                   p896_begin, p1664_begin, 0);
          LAUNCH_E4M3_WAVE_REDUCER(1664, kXQARouteLongSeqLens,
                                   p1664_launch_num_partitions, 0, p1664_begin,
                                   0, 0);
        }
#undef LAUNCH_E4M3_WAVE_REDUCER
        cuda_launch_check();

        auto& traced_wave_partitions = xqa_policy_observation(2);
        if (xqa_e4m3_g6_p64_p256_auto_trace_enabled() &&
            !traced_wave_partitions) {
          TORCH_WARN(
              "Flash-V100 XQA E4M3 G6 wave-partition route active; "
              "thresholds p256/p512/p896/p1664=",
              p256_begin, "/", p512_begin, "/", p896_begin, "/", p1664_begin,
              ", launch partitions short/p512/p896/p1664=",
              short_launch_num_partitions, "/",
              p512_route_launch_num_partitions, "/",
              p896_route_launch_num_partitions, "/",
              p1664_launch_num_partitions,
              ", merged_long=", use_merged_wave_launch,
              ", converter=shared-lut");
          traced_wave_partitions = true;
        }
        return out;
      }
      const int dual_cta_begin =
          std::max(p256_begin, xqa_e4m3_g6_dual_cta_begin());
      const int p64_launch_num_partitions =
          std::max(1, std::min(launch_num_partitions, (p256_begin + 62) / 64));
      const int p256_launch_num_partitions =
          (launch_num_partitions + 3) / 4;
      const int p256_mid_launch_num_partitions = std::max(
          1,
          std::min(p256_launch_num_partitions, (dual_cta_begin + 254) / 256));
      const int short_launch_num_partitions =
          std::max(p64_launch_num_partitions, p256_mid_launch_num_partitions);
      const int reduce_num_partitions =
          std::max(short_launch_num_partitions, p256_launch_num_partitions);

      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          0, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteShortSeqLens, false, false, false,
          KV_CACHE_DTYPE_FP8_E4M3>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          short_launch_num_partitions, false, 8, stream, dual_cta_begin,
          p256_begin, 0, false);
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<
          256, kXQARouteLongSeqLens>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          p256_launch_num_partitions, stream, false, dual_cta_begin, 0, 0,
          false);

      const dim3 reduce_grid(q.sizes()[0], q.sizes()[1], 1);
      const dim3 reduce_block(kThreadsPerBlock);
      const size_t reduce_shared_mem =
          static_cast<size_t>(2 * reduce_num_partitions) * sizeof(float);
      flash_attention_decode_reduce_kernel<256, 0, kXQARouteAllSeqLens, float>
          <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(
              tmp_out.const_data_ptr<float>(),
              max_logits.const_data_ptr<float>(),
              exp_sums.const_data_ptr<float>(),
              seq_lens.const_data_ptr<int>(),
              active_num_partitions.const_data_ptr<int>(),
              reinterpret_cast<__half*>(out.mutable_data_ptr()),
              q.sizes()[0], reduce_num_partitions, q.sizes()[1],
              tmp_out.strides()[0], tmp_out.strides()[1], tmp_out.strides()[2],
              max_logits.strides()[0], max_logits.strides()[1],
              out.strides()[0], out.strides()[1], p256_begin, 0, 0, 0);

      auto& traced_p64_p256_auto = xqa_policy_observation(3);
      if (xqa_e4m3_g6_p64_p256_auto_trace_enabled() && !traced_p64_p256_auto) {
        TORCH_WARN(
            "Flash-V100 XQA E4M3 G6 device-side p64/p256 route active; "
            "p256 threshold=",
            p256_begin, ", p64 launch partitions=", p64_launch_num_partitions,
            ", dual-CTA threshold=", dual_cta_begin,
            ", short/mid launch partitions=", short_launch_num_partitions,
            ", long p256 launch partitions=", p256_launch_num_partitions,
            ", routed launch partitions=", reduce_num_partitions);
        traced_p64_p256_auto = true;
      }
    } else if (q.sizes()[0] == 1 && partition_size == 64) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          64, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteAllSeqLens, false, false, false,
          KV_CACHE_DTYPE_FP8_E4M3>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, 8, stream);
    } else if (q.sizes()[0] == 1 && partition_size == 128) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          128, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteAllSeqLens, false, false, false,
          KV_CACHE_DTYPE_FP8_E4M3>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, 8, stream);
    } else if (q.sizes()[0] == 1 && partition_size == 512) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<512>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream, false);
    } else if (q.sizes()[0] == 1 && partition_size == 896) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<896>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream, false);
    } else if (q.sizes()[0] == 1 && partition_size == 1024) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<1024>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream, false);
    } else if (q.sizes()[0] == 1 && partition_size == 1664) {
      launch_flash_attention_decode_paged_xqa_e4m3_g6_page_dispatch<1664>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, stream, false);
    } else if (partition_size == 64) {
      LAUNCH_E4M3_BATCH_XQA(64);
    } else if (partition_size == 128) {
      LAUNCH_E4M3_BATCH_XQA(128);
    } else {
      LAUNCH_E4M3_BATCH_XQA(256);
    }
#undef LAUNCH_E4M3_BATCH_XQA
#undef LAUNCH_E4M3_DUAL_CTA
    cuda_launch_check();
    return out;
  }
  // Default only the shape with exact direct and model-level evidence.
  const bool padded_smem_enabled = xqa_padded_smem_enabled();
  const bool use_padded_smem =
      padded_smem_enabled && q_per_kv == 6 && partition_size == 256;
  const bool use_g6_dual_cta_dense = !use_padded_smem &&
                                     k_cache.sizes()[1] == 784 &&
                                     xqa_g6_dual_cta_dense_enabled();
  const bool use_g6_p1024_auto =
      q.sizes()[0] == 1 && q_per_kv == 6 && partition_size == 1024 &&
      k_cache.sizes()[1] == 784 && k_cache.sizes()[2] == 1 &&
      xqa_g6_p1024_auto_enabled();
  const bool use_g6_p1024_sawtooth =
      q.sizes()[0] == 1 && q_per_kv == 6 && partition_size == 256 &&
      k_cache.sizes()[1] == 784 && k_cache.sizes()[2] == 1 &&
      !decode_partition_size_overridden() && xqa_block784_index_enabled() &&
      xqa_g6_p1024_sawtooth_enabled();
  const bool use_g6_qk_pipeline =
      use_g6_p1024_sawtooth &&
      k_cache.scalar_type() == torch::headeronly::ScalarType::Half &&
      xqa_g6_qk_pipeline_enabled();
  const int g6_qk_pipeline_warps =
      use_g6_qk_pipeline ? xqa_g6_qk_pipeline_warps() : 0;
  const int g6_p1024_route_seq_len =
      use_g6_p1024_auto ? (get_device_prop()->multiProcessorCount + 1) * 1024
                        : 0;
  const int g6_p1024_sawtooth_p1024_mid_seq_len =
      use_g6_p1024_sawtooth ? xqa_g6_p1024_sawtooth_p1024_mid_seq_len() : 0;
  const int g6_p1024_sawtooth_p256_long_seq_len =
      use_g6_p1024_sawtooth ? xqa_g6_p1024_sawtooth_p256_long_seq_len() : 0;
  const int g6_p1024_sawtooth_p1024_final_seq_len =
      use_g6_p1024_sawtooth ? xqa_g6_p1024_sawtooth_p1024_final_seq_len() : 0;
  STD_TORCH_CHECK(
      !use_g6_p1024_sawtooth ||
          (g6_p1024_sawtooth_p1024_mid_seq_len <
               g6_p1024_sawtooth_p256_long_seq_len &&
           g6_p1024_sawtooth_p256_long_seq_len <
               g6_p1024_sawtooth_p1024_final_seq_len),
      "Invalid p1024/p256 sawtooth thresholds: ",
      g6_p1024_sawtooth_p1024_mid_seq_len, ", ",
      g6_p1024_sawtooth_p256_long_seq_len, ", ",
      g6_p1024_sawtooth_p1024_final_seq_len);
  const bool use_mtp5_dual_cta =
      q.sizes()[0] == 5 && q_per_kv == 6 && k_cache.sizes()[1] == 1616 &&
      k_cache.sizes()[2] > 0 && kv_dtype_code == KV_CACHE_DTYPE_FP8_E5M2 &&
      xqa_mtp5_dual_cta_enabled();
  const bool use_e5m2_g6_dual_cta =
      q.sizes()[0] == 1 && q_per_kv == 6 && partition_size == 256 &&
      k_cache.sizes()[1] >= 256 && k_cache.sizes()[1] % 16 == 0 &&
      k_cache.sizes()[2] == 1 &&
      k_cache.scalar_type() == torch::headeronly::ScalarType::Byte &&
      kv_dtype_code == KV_CACHE_DTYPE_FP8_E5M2 &&
      !decode_partition_size_overridden() && xqa_e5m2_g6_dual_cta_enabled() &&
      xqa_e5m2_g6_split_reduce_enabled();
  const bool batch_context_page_supported =
      k_cache.sizes()[1] == 16 ||
      (k_cache.sizes()[1] >= 256 && k_cache.sizes()[1] % 16 == 0);
  const bool use_e5m2_g6_batch_context_route =
      batch_context_max_seq_len > 0 && q.sizes()[0] >= 4 && q_per_kv == 6 &&
      batch_context_page_supported && k_cache.sizes()[2] == 1 &&
      k_cache.scalar_type() == torch::headeronly::ScalarType::Byte &&
      kv_dtype_code == KV_CACHE_DTYPE_FP8_E5M2 &&
      !decode_partition_size_overridden();
  const XQABatchContextRoute batch_context_route =
      use_e5m2_g6_batch_context_route
          ? select_xqa_batch_context_route(q.sizes()[0],
                                           batch_context_max_seq_len,
                                           partition_size)
          : XQABatchContextRoute::kDisabled;
  const int e5m2_g6_dual_cta_seq_len =
      use_e5m2_g6_dual_cta
          ? get_device_prop()->multiProcessorCount * 1024 + 1
          : 0;
  const int e5m2_p1024_begin =
      use_e5m2_g6_dual_cta ? xqa_e5m2_p1024_begin() : 0;
  const int e5m2_scalar_xqa_seq_len =
      use_e5m2_g6_dual_cta
          ? std::min(xqa_e5m2_scalar_xqa_seq_len(), e5m2_p1024_begin)
          : 0;
  STD_TORCH_CHECK(
      !use_e5m2_g6_dual_cta || e5m2_p1024_begin < e5m2_g6_dual_cta_seq_len,
      "Invalid E5M2 G6 partition thresholds: ", e5m2_p1024_begin, ", ",
      e5m2_g6_dual_cta_seq_len);
  const int e5m2_p1024_launch_num_partitions =
      (launch_num_partitions + 3) / 4;
  const int e5m2_p1024_one_cta_launch_num_partitions =
      use_e5m2_g6_dual_cta
          ? std::min(e5m2_p1024_launch_num_partitions,
                     get_device_prop()->multiProcessorCount)
          : 0;
  const bool use_e5m2_partition_page_ids =
      use_e5m2_g6_dual_cta && xqa_e5m2_partition_page_ids_enabled();
  const bool use_e5m2_pair_load =
      use_e5m2_partition_page_ids && xqa_e5m2_pair_load_enabled();
  const bool use_e5m2_batch_wide_load =
      use_e5m2_g6_batch_context_route && partition_size == 256 &&
      k_cache.sizes()[1] >= 256 && k_cache.sizes()[1] % 16 == 0 &&
      (batch_context_route == XQABatchContextRoute::kDualCta ||
       batch_context_route == XQABatchContextRoute::kDualCtaSplit) &&
      xqa_e5m2_batch_wide_load_enabled();
  const bool use_qsa_page4 =
      q.sizes()[0] >= 4096 && q_per_kv == 6 &&
      (partition_size == 256 || partition_size == 512 ||
       partition_size == 1024) &&
      k_cache.sizes()[1] == 4 && k_cache.sizes()[2] == 1 &&
      k_cache.scalar_type() == torch::headeronly::ScalarType::Half &&
      block_table.sizes()[1] == 513;
  const bool use_g6_dual_cta =
      use_qsa_page4 || use_g6_p1024_auto || use_g6_p1024_sawtooth ||
      use_mtp5_dual_cta || use_e5m2_g6_dual_cta ||
      batch_context_route == XQABatchContextRoute::kDualCta ||
      batch_context_route == XQABatchContextRoute::kDualCtaSplit ||
      (xqa_g6_dual_cta_enabled() && (use_padded_smem || use_g6_dual_cta_dense));
  const bool use_split_reduce =
      use_g6_p1024_auto || use_g6_p1024_sawtooth || use_e5m2_g6_dual_cta ||
      batch_context_route == XQABatchContextRoute::kDualCtaSplit ||
      (use_g6_dual_cta && xqa_split_reduce_enabled());
  const bool supports_block16_index =
      use_g6_dual_cta && k_cache.sizes()[1] == 16;
  const bool use_block4_index = use_g6_dual_cta && k_cache.sizes()[1] == 4;
  const bool supports_block16_contiguous_layout =
      supports_block16_index && k_cache.sizes()[2] == 1 &&
      k_cache.strides()[0] == 4096 && k_cache.strides()[1] == 256 &&
      k_cache.strides()[2] == 256 && k_cache.strides()[3] == 1 &&
      v_cache.strides()[0] == 4096 && v_cache.strides()[1] == 256 &&
      v_cache.strides()[2] == 256 && v_cache.strides()[3] == 1;
  const bool use_block784_index =
      use_g6_dual_cta && k_cache.sizes()[1] == 784 &&
      xqa_block784_index_enabled();
  const bool use_aligned_padded_smem =
      use_padded_smem && use_block784_index &&
      xqa_aligned_padded_smem_enabled();
  const int requested_block16_layout_mode = xqa_block16_layout_mode();
  const int block16_layout_mode =
      requested_block16_layout_mode == 1 && supports_block16_index
          ? 1
          : requested_block16_layout_mode == 2 &&
                    supports_block16_contiguous_layout
                ? 2
                : 0;
  if (xqa_block16_layout_required() && requested_block16_layout_mode != 0) {
    STD_TORCH_CHECK(
        block16_layout_mode == requested_block16_layout_mode,
        "Requested block16 XQA mode ", requested_block16_layout_mode,
        " but the live KV cache does not satisfy its layout gate: block_size=",
        k_cache.sizes()[1], ", num_kv_heads=", k_cache.sizes()[2],
        ", k_strides=[", k_cache.strides()[0], ",", k_cache.strides()[1], ",",
        k_cache.strides()[2], ",", k_cache.strides()[3], "]");
  }
  trace_xqa_batch_context_route(q.sizes()[0], batch_context_max_seq_len,
                                partition_size, k_cache.sizes()[1],
                                batch_context_route);
  auto& traced_e5m2_batch_wide_load = xqa_policy_observation(4);
  if (xqa_batch_context_routing_trace_enabled() && use_e5m2_batch_wide_load &&
      !traced_e5m2_batch_wide_load) {
    TORCH_WARN(
        "Flash-V100 XQA batched E5M2 partition page reuse and paired "
        "128-bit loads active");
    traced_e5m2_batch_wide_load = true;
  }
  auto& traced_block16_layout = xqa_policy_observation(5);
  if (xqa_block16_layout_trace_enabled() && block16_layout_mode != 0 &&
      !traced_block16_layout) {
    TORCH_WARN("Flash-V100 XQA block16 mode ", block16_layout_mode,
               " active for KV shape [blocks,", k_cache.sizes()[1], ",",
               k_cache.sizes()[2], ",", k_cache.sizes()[3], "]");
    traced_block16_layout = true;
  }
  auto& traced_block784_index = xqa_policy_observation(6);
  if (xqa_block784_index_trace_enabled() && use_block784_index &&
      !traced_block784_index) {
    TORCH_WARN(
        "Flash-V100 XQA block784 index specialization active for KV "
        "shape [blocks,",
        k_cache.sizes()[1], ",", k_cache.sizes()[2], ",", k_cache.sizes()[3],
        "]");
    traced_block784_index = true;
  }
  auto& traced_g6_p1024_auto = xqa_policy_observation(7);
  if (xqa_g6_p1024_auto_trace_enabled() && use_g6_p1024_auto &&
      !traced_g6_p1024_auto) {
    TORCH_WARN(
        "Flash-V100 XQA p1024 dynamic one/two-CTA route active; "
        "long-sequence threshold=",
        g6_p1024_route_seq_len);
    traced_g6_p1024_auto = true;
  }
  auto& traced_g6_p1024_sawtooth = xqa_policy_observation(8);
  if (xqa_g6_p1024_sawtooth_trace_enabled() && use_g6_p1024_sawtooth &&
      !traced_g6_p1024_sawtooth) {
    TORCH_WARN("Flash-V100 XQA p1024/p256 sawtooth route active; thresholds=",
               g6_p1024_sawtooth_p1024_mid_seq_len, ", ",
               g6_p1024_sawtooth_p256_long_seq_len, ", ",
               g6_p1024_sawtooth_p1024_final_seq_len);
    traced_g6_p1024_sawtooth = true;
  }
  auto& traced_g6_qk_pipeline = xqa_policy_observation(9);
  if (xqa_g6_qk_pipeline_trace_enabled() && use_g6_qk_pipeline &&
      !traced_g6_qk_pipeline) {
    TORCH_WARN(
        "Flash-V100 XQA G6 fp16-KV QK K64 producer/consumer "
        "pipeline active for the mid p1024 range with baseline final-range "
        "fallback; warps=",
        g6_qk_pipeline_warps);
    traced_g6_qk_pipeline = true;
  }
  auto& traced_e5m2_g6_dual_cta = xqa_policy_observation(10);
  if (xqa_e5m2_g6_dual_cta_trace_enabled() && use_e5m2_g6_dual_cta &&
      !traced_e5m2_g6_dual_cta) {
    TORCH_WARN(
        "Flash-V100 XQA E5M2 G6 device-side p256/p1024 route active; "
        "thresholds=",
        e5m2_p1024_begin, ", ", e5m2_g6_dual_cta_seq_len, ", KV shape [blocks,",
        k_cache.sizes()[1], ",", k_cache.sizes()[2], ",", k_cache.sizes()[3],
        "], split_reduce=", use_split_reduce,
        ", partition_page_ids=", use_e5m2_partition_page_ids,
        ", pair_load=", use_e5m2_pair_load,
        ", scalar_xqa_seq_len=", e5m2_scalar_xqa_seq_len);
    traced_e5m2_g6_dual_cta = true;
  }
  auto& traced_aligned_padded_smem = xqa_policy_observation(11);
  if (xqa_aligned_padded_smem_trace_enabled() && use_aligned_padded_smem &&
      !traced_aligned_padded_smem) {
    TORCH_WARN("Flash-V100 XQA aligned padded shared layout active");
    traced_aligned_padded_smem = true;
  }
  const int split_reduce_dim_tile = xqa_split_reduce_dim_tile();

// KV_DTYPE_OVERRIDE is -1 (the launcher's runtime fp16/fp8 selection) for
// fp16/fp8, or KV_CACHE_DTYPE_NVFP4 to pin the pool-geometry load at
// compile time. The explicitly spelled template arguments equal the
// launcher defaults.
#define LAUNCH_XQA_WIDE(GROUP_SIZE, PARTITION, KV_DTYPE_OVERRIDE)              \
  do {                                                                         \
    if (use_padded_smem) {                                                     \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<                     \
          PARTITION, GROUP_SIZE, true, kXQATC256WideThreads, 1, 0, false,      \
          false, kXQARouteAllSeqLens, false, false, false,                     \
          KV_DTYPE_OVERRIDE>(                                                  \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale,          \
          k_scale, v_scale, launch_num_partitions, false, 8, stream);          \
    } else {                                                                   \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<                     \
          PARTITION, GROUP_SIZE, false, kXQATC256WideThreads, 1, 0, false,     \
          false, kXQARouteAllSeqLens, false, false, false,                     \
          KV_DTYPE_OVERRIDE>(                                                   \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale,          \
          k_scale, v_scale, launch_num_partitions, false, 8, stream);          \
    }                                                                          \
  } while (0)

#define DISPATCH_PARTITION(GROUP_SIZE, KV_DTYPE_OVERRIDE)                      \
  do {                                                                        \
    switch (partition_size) {                                                \
      case 256:                                                               \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 256, KV_DTYPE_OVERRIDE);                  \
        break;                                                                \
      case 128:                                                               \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 128, KV_DTYPE_OVERRIDE);                  \
        break;                                                                \
      case 64:                                                                \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 64, KV_DTYPE_OVERRIDE);                   \
        break;                                                                \
      case 32:                                                                \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 32, KV_DTYPE_OVERRIDE);                   \
        break;                                                                \
      case 16:                                                                \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 16, KV_DTYPE_OVERRIDE);                   \
        break;                                                                \
      case 512:                                                               \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 512, KV_DTYPE_OVERRIDE);                  \
        break;                                                                \
      case 1024:                                                              \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 1024, KV_DTYPE_OVERRIDE);                 \
        break;                                                                \
      default:                                                                \
        STD_TORCH_CHECK(                                                      \
            false, "Unsupported XQA partition_size: ", partition_size);       \
    }                                                                         \
  } while (0)

  if (use_e5m2_g6_dual_cta) {
    launch_flash_attention_decode_paged<
        256, 256, KV_CACHE_DTYPE_FP8_E5M2, kXQARouteShortSeqLens>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, launch_num_partitions,
        k_scale, v_scale, window_size_left, window_size_right, stream,
        e5m2_scalar_xqa_seq_len, 0, 0, false);
    if (use_e5m2_pair_load) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid, false, true, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, split_reduce_dim_tile, stream,
          e5m2_scalar_xqa_seq_len, e5m2_p1024_begin, 0, false);
    } else if (use_e5m2_partition_page_ids) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid, false, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, split_reduce_dim_tile, stream,
          e5m2_scalar_xqa_seq_len, e5m2_p1024_begin, 0, false);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, false, split_reduce_dim_tile, stream,
          e5m2_scalar_xqa_seq_len, e5m2_p1024_begin, 0, false);
    }
    if (use_e5m2_partition_page_ids) {
      if (use_e5m2_pair_load) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATC256WideThreads, 1, 0, false, false,
            kXQARouteP1024SawtoothMid, false, true, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_one_cta_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_p1024_begin,
            e5m2_g6_dual_cta_seq_len, 0, false);
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false, false,
            kXQARouteLongSeqLens, false, true, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_g6_dual_cta_seq_len, 0, 0,
            false);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATC256WideThreads, 1, 0, false, false,
            kXQARouteP1024SawtoothMid, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_one_cta_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_p1024_begin,
            e5m2_g6_dual_cta_seq_len, 0, false);
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false, false,
            kXQARouteLongSeqLens, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, e5m2_p1024_launch_num_partitions, false,
            split_reduce_dim_tile, stream, e5m2_g6_dual_cta_seq_len, 0, 0,
            false);
      }
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATC256WideThreads, 1, 0, false, false,
          kXQARouteP1024SawtoothMid>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          e5m2_p1024_one_cta_launch_num_partitions, false,
          split_reduce_dim_tile, stream, e5m2_p1024_begin,
          e5m2_g6_dual_cta_seq_len, 0, false);
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false, false,
          kXQARouteLongSeqLens>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          e5m2_p1024_launch_num_partitions, false, split_reduce_dim_tile,
          stream, e5m2_g6_dual_cta_seq_len, 0, 0, false);
    }
    launch_flash_attention_decode_xqa_split_reduce<0>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream, e5m2_p1024_begin,
        e5m2_g6_dual_cta_seq_len, e5m2_g6_dual_cta_seq_len);
  } else if (use_g6_p1024_sawtooth) {
    const int p1024_launch_num_partitions = (launch_num_partitions + 3) / 4;
    if (use_g6_qk_pipeline) {
      if (g6_qk_pipeline_warps == 8) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6Pipeline8WarpThreads, 2, 784, false, false,
            kXQARouteP1024SawtoothMid, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, p1024_launch_num_partitions, false, split_reduce_dim_tile,
            stream, g6_p1024_sawtooth_p1024_mid_seq_len,
            g6_p1024_sawtooth_p256_long_seq_len,
            g6_p1024_sawtooth_p1024_final_seq_len, false);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
            kXQARouteP1024SawtoothMid, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, p1024_launch_num_partitions, false, split_reduce_dim_tile,
            stream, g6_p1024_sawtooth_p1024_mid_seq_len,
            g6_p1024_sawtooth_p256_long_seq_len,
            g6_p1024_sawtooth_p1024_final_seq_len, false);
      }
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
          kXQARouteP1024SawtoothFinal>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          p1024_launch_num_partitions, false, split_reduce_dim_tile, stream,
          g6_p1024_sawtooth_p1024_mid_seq_len,
          g6_p1024_sawtooth_p256_long_seq_len,
          g6_p1024_sawtooth_p1024_final_seq_len, false);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
          kXQARouteP1024Sawtooth>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          p1024_launch_num_partitions, false, split_reduce_dim_tile, stream,
          g6_p1024_sawtooth_p1024_mid_seq_len,
          g6_p1024_sawtooth_p256_long_seq_len,
          g6_p1024_sawtooth_p1024_final_seq_len, false);
    }
    launch_flash_attention_decode_paged_xqa_tc_256_wide<
        256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false, false,
        kXQARouteP256Sawtooth>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
        launch_num_partitions, false, split_reduce_dim_tile, stream,
        g6_p1024_sawtooth_p1024_mid_seq_len,
        g6_p1024_sawtooth_p256_long_seq_len,
        g6_p1024_sawtooth_p1024_final_seq_len, false);
    launch_flash_attention_decode_xqa_split_reduce<0>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream, g6_p1024_sawtooth_p1024_mid_seq_len,
        g6_p1024_sawtooth_p256_long_seq_len,
        g6_p1024_sawtooth_p1024_final_seq_len);
  } else if (use_g6_p1024_auto) {
    launch_flash_attention_decode_paged_xqa_tc_256_wide<
        1024, 6, false, kXQATC256WideThreads, 1, 784, false, false,
        kXQARouteShortSeqLens>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
        launch_num_partitions, false, split_reduce_dim_tile, stream,
        g6_p1024_route_seq_len, 0, 0, false);
    launch_flash_attention_decode_paged_xqa_tc_256_wide<
        1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false, false,
        kXQARouteLongSeqLens>(
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
        exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
        launch_num_partitions, true, split_reduce_dim_tile, stream,
        g6_p1024_route_seq_len, 0, 0, true);
  } else if (use_g6_dual_cta) {
    if (use_block784_index) {
      if (partition_size == 1024) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (partition_size == 512) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            512, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (use_aligned_padded_smem) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (use_padded_smem) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (use_block4_index) {
      if (partition_size == 256) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 4, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (partition_size == 512) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            512, 6, true, kXQATCG6DualCtaThreads, 2, 4, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            1024, 6, true, kXQATCG6DualCtaThreads, 2, 4, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (block16_layout_mode == 2) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 16, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (block16_layout_mode == 1) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 16, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (partition_size == 256) {
      if (use_e5m2_batch_wide_load) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 0, false, false,
            kXQARouteAllSeqLens, false, true, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 0, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (partition_size == 512) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          512, 6, false, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    }
  } else {
    // NVFP4 takes only the generic tc-256-wide route (the G6 dual-CTA
    // specializations carry their own KV geometry assumptions). The
    // dtype override must be a constant expression, so the runtime
    // branch pins it per arm: -1 = the launcher's runtime fp16/fp8
    // selection (original behavior), NVFP4 = the pool-geometry load.
    STD_TORCH_CHECK(
        kv_dtype_code != KV_CACHE_DTYPE_NVFP4 || !use_g6_dual_cta,
        "NVFP4 XQA does not support the active G6 dual-CTA specialization");
    if (kv_dtype_code == KV_CACHE_DTYPE_NVFP4) {
      if (q_per_kv == 4) {
        DISPATCH_PARTITION(4, KV_CACHE_DTYPE_NVFP4);
      } else if (q_per_kv == 6) {
        DISPATCH_PARTITION(6, KV_CACHE_DTYPE_NVFP4);
      } else {
        DISPATCH_PARTITION(8, KV_CACHE_DTYPE_NVFP4);
      }
    } else if (q_per_kv == 4) {
      DISPATCH_PARTITION(4, -1);
    } else if (q_per_kv == 6) {
      DISPATCH_PARTITION(6, -1);
    } else {
      DISPATCH_PARTITION(8, -1);
    }
  }

#undef DISPATCH_PARTITION
#undef LAUNCH_XQA_WIDE

  cuda_launch_check();
  return out;
}

#undef TORCH_WARN

}  // namespace qsa_page4
}  // namespace vllm
