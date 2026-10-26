// SM70 NVFP4 fused dequant -> dense FP16 (single pass).
//
// The SM70 NVFP4 linear kernel dequants the checkpoint's NVFP4 weight to a dense
// FP16 weight every forward (for the n%64==0 layers that cannot be cached in
// FP16 on a 32GB V100). It fuses the whole thing into one memory-bound pass:
// read the row-major packed NVFP4 codes + the per-16-group FP8-E4M3 scales,
// decode, and write the dense FP16 weight with the per-tensor global scale
// baked in (the layer stores weight_global_scale as a multiplier, the
// reciprocal of the checkpoint's dequant divisor). The layout matches the
// checkpoint's natural packing (low nibble = even k, high nibble = odd k; one
// scale per 16-wide K group), so no repack is needed.
//
// Decode: E2M1 (FP4) is split into its 2-bit exponent (E2: 4 distinct bases)
// and 1-bit mantissa (M1: 2 distinct multipliers), so the per-code gather hits
// only 4 addresses instead of 8/16 for a flat code LUT -- half the warp
// divergence. All decode math runs in FP16 (the output is FP16 and the
// downstream GEMM is FP16), which on V100 is ~1.4x faster than an FP32 decode
// (467 vs 320 GB/s) at the same store width.
//
// Registers op `sm70_nvfp4_dequant(weight, weight_scale, global_scale) -> fp16 (n, k)`.

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace {

// E2M1 (FP4): 1 sign + 2-bit exp (E2) + 1-bit mant (M1) -> half (all E2M1
// magnitudes are exactly representable in half). E2=0 is the subnormal bin
// (M1 ? .5 : 0); E2>=1 gives base 2^(E2-1) * (M1 ? 1.5 : 1.0). The 4-entry
// base gather is the low-divergence E2 part of the split.
__device__ __forceinline__ __half nvfp4_e2m1_to_half(uint8_t code) {
  const int E2 = (code >> 1) & 3;
  const int M1 = code & 1;
  __half v;
  if (E2 == 0) {
    v = M1 ? __float2half(0.5f) : __half(0);  // subnormal bin
  } else {
    const __half base = (E2 == 1) ? __half(1) : (E2 == 2) ? __half(2) : __half(4);
    v = __hmul(base, M1 ? __float2half(1.5f) : __half(1));
  }
  return (code & 0x8) ? __hneg(v) : v;  // sign bit [3]
}

// FP8-E4M3 -> half (exactly representable; computed in FP32 then rounded once
// to half). Called once per 16-wide group, so the FP32 cost is amortized.
__device__ __forceinline__ __half fp8_e4m3_to_half(uint8_t b) {
  const int sgn = (b >> 7) & 1;
  const int e = (b >> 3) & 0xF;
  const int m = b & 0x7;
  const float v = (e == 0) ? (float(m) * (1.0f / 512.0f))
                           : ((1.0f + float(m) / 8.0f) * exp2f(float(e) - 7.0f));
  return __float2half(sgn ? -v : v);
}

// Effective group scale in one rounding: the exact E4M3 group scale times the
// FP32 global scale. Rounding the global factor to FP16 first and then
// multiplying in FP16 changes the dequantised weights (and disagrees with the
// QPN2 decode kernel's folding), so prefill and decode would see different
// values for the same layer.
__device__ __forceinline__ __half nvfp4_effective_scale_h(uint8_t value,
                                                         float global_scale) {
  return __float2half_rn(
      __fmul_rn(__half2float(fp8_e4m3_to_half(value)), global_scale));
}

__global__ void nvfp4_dequant_fused_kernel(
    const uint8_t* __restrict__ weight,  // (n, k/2) row-major
    const uint8_t* __restrict__ scale,  // (n, k/16) row-major fp8
    float global_scale, int n, int k,
    __half* __restrict__ out) {  // (n, k)
  // One thread per 16-wide K group: 1 scale byte, 8 weight bytes (16 codes),
  // 32-byte-aligned output. Memory-bound -> vectorize the load + store.
  const int ngroups = k >> 4;
  const int total = n * ngroups;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int i = idx / ngroups;
  const int g = idx - i * ngroups;
  const int kstart = g << 4;
  // k is a multiple of 16, so the 8-byte (uint2) weight load is aligned.
  const uint2 wd = *reinterpret_cast<const uint2*>(weight + (size_t)i * (k >> 1) +
                                                    (kstart >> 1));
  const uint32_t* pw = reinterpret_cast<const uint32_t*>(&wd);
  const __half h0 = nvfp4_effective_scale_h(
      scale[(size_t)i * ngroups + g], global_scale);
  __half outv[16];
#pragma unroll
  for (int p = 0; p < 16; ++p) {
    const uint8_t byte = (pw[(p >> 1) / 4] >> (((p >> 1) % 4) * 8)) & 0xFF;
    const int code = ((p & 1) ? (byte >> 4) : (byte & 0xF));
    outv[p] = __hmul(nvfp4_e2m1_to_half(code), h0);
  }
  // Store the 16 halves as two 16-byte (128-bit) stores.
  float4* op16 = reinterpret_cast<float4*>(out + (size_t)i * k + kstart);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    float4 fv;
    *reinterpret_cast<__half2*>(&fv.x) =
        __half2(outv[8 * q + 0], outv[8 * q + 1]);
    *reinterpret_cast<__half2*>(&fv.y) =
        __half2(outv[8 * q + 2], outv[8 * q + 3]);
    *reinterpret_cast<__half2*>(&fv.z) =
        __half2(outv[8 * q + 4], outv[8 * q + 5]);
    *reinterpret_cast<__half2*>(&fv.w) =
        __half2(outv[8 * q + 6], outv[8 * q + 7]);
    op16[q] = fv;
  }
}

// Inverse of the QPN2 code/scale repack (see sm70_nvfp4_qpn2.cu) so a
// natural-order dequant of the QPN2 layout lands each code at its (row, k)
// output slot. kQpn2InvCol[col_in_tile] = lane; kQpn2InvKPhys[k_in_group] =
// physical code position (0..15; physical_byte = pos>>1, half = pos&1).
__constant__ int kQpn2InvCol[32] = {
    0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23,
    8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
__constant__ int kQpn2InvKPhys[16] = {
    0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15};

// Dequant the QPN2-repacked 4-bit weight (n,k/2 codes + n,k/16 e4m3 scales) to
// a dense natural-row-major FP16 weight (n,k) with global_scale baked in. This
// is the 1x prefill dequant: the single 4-bit storage IS the QPN2 layout
// (shared with the decode GEMM); prefill dequants it to FP16 then runs a plain
// FP16 GEMM. Natural-order iteration keeps the output stores coalesced.
// One thread per 16-wide K group: the 16 output codes all live in one 8-byte
// QPN2 lane word (lane = kQpn2InvCol[row&31]), so a single aligned uint2 load
// fetches them; the phys table reorders nibble->kcol. The 16 halves are stored
// as two 128-bit (float4) stores, coalesced across the warp. This vectorized
// store is ~2x faster than a scalar per-element store (write is the dominant
// HBM traffic); the scattered QPN2 read stays L2-resident (a 32-row tile is
// ~80KB) so it does not limit bandwidth.
__global__ void nvfp4_dequant_qpn2_kernel(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales,
    float global_scale, int n, int k,
    __half* __restrict__ out) {
  const int groups_k16 = k >> 4;
  const int total = n * groups_k16;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int row = idx / groups_k16;
  const int group = idx - row * groups_k16;
  const int lane = kQpn2InvCol[row & 31];
  const size_t base = (size_t)(row >> 5) * groups_k16 + group;
  const uint2 wd =
      *reinterpret_cast<const uint2*>(codes + base * 256 + lane * 8);
  const uint32_t* pw = reinterpret_cast<const uint32_t*>(&wd);
  const __half h0 =
      nvfp4_effective_scale_h(scales[base * 32 + lane], global_scale);
  __half outv[16];
#pragma unroll
  for (int p = 0; p < 16; ++p) {
    const int phys = kQpn2InvKPhys[p];
    const uint8_t byte =
        (pw[(phys >> 1) / 4] >> (((phys >> 1) % 4) * 8)) & 0xFF;
    const int code = (phys & 1) ? (byte >> 4) : (byte & 0xF);
    outv[p] = __hmul(nvfp4_e2m1_to_half(code), h0);
  }
  float4* op16 =
      reinterpret_cast<float4*>(out + (size_t)row * k + group * 16);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    float4 fv;
    *reinterpret_cast<__half2*>(&fv.x) = __half2(outv[8*q+0], outv[8*q+1]);
    *reinterpret_cast<__half2*>(&fv.y) = __half2(outv[8*q+2], outv[8*q+3]);
    *reinterpret_cast<__half2*>(&fv.z) = __half2(outv[8*q+4], outv[8*q+5]);
    *reinterpret_cast<__half2*>(&fv.w) = __half2(outv[8*q+6], outv[8*q+7]);
    op16[q] = fv;
  }
}

inline void nvfp4_dequant_check_launch() {
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "NVFP4 dequant launch failed: ",
                  cudaGetErrorString(err));
}

}  // namespace

torch::stable::Tensor sm70_nvfp4_dequant(
    torch::stable::Tensor& weight, torch::stable::Tensor& weight_scale,
    double global_scale) {
  torch::stable::accelerator::DeviceGuard device_guard(
      weight.get_device_index());
  int64_t n = weight.size(0);
  int64_t k = weight.size(1) * 2;
  std::vector<int64_t> out_shape = {n, k};
  torch::headeronly::IntHeaderOnlyArrayRef out_sizes(out_shape.data(),
                                                     out_shape.size());
  // The kernel writes every output element (one thread per 16-wide K group),
  // so the allocation need not be zeroed -- new_zeros would add a redundant
  // full-output memset (~the same traffic as the kernel itself) that halved the
  // effective dequant bandwidth.
  auto out =
      torch::stable::new_empty(weight, out_sizes, torch::headeronly::ScalarType::Half);

  int64_t const total = n * (k / 16);
  const int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = get_current_cuda_stream(weight.get_device_index());
  nvfp4_dequant_fused_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight.data_ptr()),
      reinterpret_cast<const uint8_t*>(weight_scale.data_ptr()),
      static_cast<float>(global_scale), static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<__half*>(out.mutable_data_ptr()));
  nvfp4_dequant_check_launch();
  return out;
}

torch::stable::Tensor sm70_nvfp4_dequant_from_qpn2(
    torch::stable::Tensor& codes, torch::stable::Tensor& scales,
    double global_scale) {
  torch::stable::accelerator::DeviceGuard device_guard(
      codes.get_device_index());
  int64_t n = codes.size(0);
  int64_t k = codes.size(1) * 2;
  std::vector<int64_t> out_shape = {n, k};
  torch::headeronly::IntHeaderOnlyArrayRef out_sizes(out_shape.data(),
                                                     out_shape.size());
  auto out =
      torch::stable::new_empty(codes, out_sizes, torch::headeronly::ScalarType::Half);
  int64_t const total = n * (k >> 4);
  const int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  cudaStream_t stream = get_current_cuda_stream(codes.get_device_index());
  nvfp4_dequant_qpn2_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(codes.data_ptr()),
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<float>(global_scale), static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<__half*>(out.mutable_data_ptr()));
  nvfp4_dequant_check_launch();
  return out;
}

// Dequant the QPN2 layout into a pre-allocated output buffer (no allocation).
// Used by the dispatch op with a persistent scratch to avoid per-call new_empty.
void sm70_nvfp4_dequant_from_qpn2_into(
    torch::stable::Tensor& codes, torch::stable::Tensor& scales,
    double global_scale, torch::stable::Tensor& out,
    cudaStream_t stream = nullptr) {
  torch::stable::accelerator::DeviceGuard device_guard(
      codes.get_device_index());
  if (!stream)
    stream = get_current_cuda_stream(codes.get_device_index());
  int64_t n = codes.size(0);
  int64_t k = codes.size(1) * 2;
  int64_t const total = n * (k >> 4);
  const int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  nvfp4_dequant_qpn2_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(codes.data_ptr()),
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<float>(global_scale), static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<__half*>(out.mutable_data_ptr()));
  nvfp4_dequant_check_launch();
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nvfp4_dequant_ops) {
  sm70_nvfp4_dequant_ops.def(
      "sm70_nvfp4_dequant(Tensor weight, Tensor weight_scale, "
      "float global_scale) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nvfp4_dequant_ops) {
  sm70_nvfp4_dequant_ops.impl("sm70_nvfp4_dequant",
                              TORCH_BOX(&sm70_nvfp4_dequant));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nvfp4_dequant_qpn2_ops) {
  sm70_nvfp4_dequant_qpn2_ops.def(
      "sm70_nvfp4_dequant_from_qpn2(Tensor codes, Tensor scales, "
      "float global_scale) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nvfp4_dequant_qpn2_ops) {
  sm70_nvfp4_dequant_qpn2_ops.impl("sm70_nvfp4_dequant_from_qpn2",
                                   TORCH_BOX(&sm70_nvfp4_dequant_from_qpn2));
}
