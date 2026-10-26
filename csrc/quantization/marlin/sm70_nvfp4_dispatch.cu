// SM70 NVFP4 dispatch op - a single torch op that selects the QPN2 decode
// GEMM (m <= 32) or the dequant->FP16 + cuBLAS prefill GEMM (m > 32) internally.

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <mutex>
#include <vector>

// Defined in sm70_nvfp4_qpn2.cu / sm70_nvfp4_dequant.cu (same .so, global ns).
void sm70_nvfp4_qpn2_gemm(torch::stable::Tensor& out,
                          torch::stable::Tensor& input,
                          torch::stable::Tensor& codes,
                          torch::stable::Tensor& scales, double global_scale,
                          int64_t split_k, int64_t accumulator_chains);
void sm70_nvfp4_dequant_from_qpn2_into(
    torch::stable::Tensor& codes, torch::stable::Tensor& scales,
    double global_scale, torch::stable::Tensor& out,
    cudaStream_t stream = nullptr);

namespace {
constexpr int64_t kQpn2MaxM = 32;

struct Sm70PrefillState {
  torch::stable::Tensor buf;   // persistent FP16 scratch (reused across calls)
  cublasHandle_t handle = nullptr;
  bool initialized = false;
};

// One state per device (lazy init on first prefill dispatch).
Sm70PrefillState& sm70_state(int device) {
  static std::vector<Sm70PrefillState*> states;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (static_cast<int>(device) >= static_cast<int>(states.size())) {
    states.resize(device + 1, nullptr);
  }
  if (!states[device]) {
    auto* s = new Sm70PrefillState();
    cudaSetDevice(device);
    cublasCreate(&s->handle);
    s->initialized = true;
    states[device] = s;
  }
  return *states[device];
}

// Grow the persistent scratch to (n, k) if needed.
void ensure_buf(Sm70PrefillState& s, torch::stable::Tensor& ref,
                int64_t n, int64_t k) {
  if (!s.buf.defined() || s.buf.size(0) * s.buf.size(1) < n * k) {
    std::vector<int64_t> shape = {n, k};
    torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(), shape.size());
    s.buf = torch::stable::new_empty(ref, sizes,
                                      torch::headeronly::ScalarType::Half);
  }
}
}  // namespace

void sm70_nvfp4_gemm_dispatch(torch::stable::Tensor& out,
                              torch::stable::Tensor& input,
                              torch::stable::Tensor& qpn2_codes,
                              torch::stable::Tensor& qpn2_scales,
                              double qpn2_global_scale, int64_t split_k,
                              int64_t accumulator_chains) {
  const int64_t m = input.size(0);
  const int64_t n = out.size(1);
  const int64_t k = input.size(1);
  if (m <= kQpn2MaxM) {
    sm70_nvfp4_qpn2_gemm(out, input, qpn2_codes, qpn2_scales,
                         qpn2_global_scale, split_k, accumulator_chains);
    return;
  }
  const int device = input.get_device_index();
  const cudaStream_t stream = get_current_cuda_stream(device);
  auto& st = sm70_state(device);
  ensure_buf(st, qpn2_codes, n, k);

  // Dequant the QPN2 layout into the persistent scratch, then run the FP16
  // GEMM, both on the current stream. (A ping-pong side-stream overlap was
  // tried and gives no E2E gain on SM70 -- the GPU does not co-schedule a
  // memory-bound dequant with a compute-bound GEMM -- so it is run serially.)
  sm70_nvfp4_dequant_from_qpn2_into(qpn2_codes, qpn2_scales,
                                    qpn2_global_scale, st.buf, stream);
  const __half* a = reinterpret_cast<const __half*>(input.data_ptr());
  const __half* w = reinterpret_cast<const __half*>(st.buf.data_ptr());
  __half* c = reinterpret_cast<__half*>(out.mutable_data_ptr());
  const float alpha = 1.0f, beta = 0.0f;
  cublasSetStream(st.handle, stream);
  cublasGemmEx(st.handle, CUBLAS_OP_T, CUBLAS_OP_N,
               static_cast<int>(n), static_cast<int>(m),
               static_cast<int>(k), &alpha,
               w, CUDA_R_16F, static_cast<int>(k),
               a, CUDA_R_16F, static_cast<int>(k),
               &beta, c, CUDA_R_16F, static_cast<int>(n),
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nvfp4_dispatch_ops) {
  sm70_nvfp4_dispatch_ops.def(
      "sm70_nvfp4_gemm_dispatch(Tensor(a!) out, Tensor input, Tensor qpn2_codes, "
      "Tensor qpn2_scales, float qpn2_global_scale, int split_k, "
      "int nacc) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nvfp4_dispatch_ops) {
  sm70_nvfp4_dispatch_ops.impl("sm70_nvfp4_gemm_dispatch",
                              TORCH_BOX(&sm70_nvfp4_gemm_dispatch));
}
