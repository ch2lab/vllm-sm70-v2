// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// SM70 (V100) plan-free non-causal FP16 paged attention for the DFlash2
// draft model.
//
// The draft's KV is small and bounded (its own accumulated context, ~1e3
// tokens — NOT the target's 132K context), so split-KV is unnecessary: one
// CTA per (kv head, request) streams the full KV and does a single-pass
// non-causal online-softmax attention. Cudagraph-safe (fixed grid, no
// seq_lens CPU sync, no FA2 plan, no NVFP4 dequant).
//
//   partition kernel  grid = (H_kv, R), block = kRatio*32
//     one (kv head, request) per CTA; warp = qo head within the kv group,
//     so the staged KV is read exactly once per kv head (1x traffic, the
//     kRatio qo heads share it). Non-causal: every q row attends to
//     [0, kv_len). sm_scale is applied to the raw score once, and the
//     running max / denominator are tracked in that scaled space.
//
// KV layout (HND, K/V interleaved in the last dim, as produced by
// FlashInferImpl.forward's kv_cache_permute):
//   [num_blocks, H_kv, block_size, 2*head_dim] fp16
//   K: kv[page, h, offset, d]            for d in [0, head_dim)
//   V: kv[page, h, offset, head_dim + d] for d in [0, head_dim)
// Strides passed at launch: kv_s0 (page), kv_sh (head), kv_soff (offset);
// the last dim is contiguous (stride 1).

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

// Head geometry for the draft (per rank, TP4): kH=2 kv heads, kHQ=8 qo heads.
constexpr int kD = 128;
constexpr int kN = 16;  // tokens staged per chunk
constexpr int kMaxQ = 8;
constexpr int kLaneDims = 4;  // 128 dims / 32 lanes

// One CTA per (kv head, request). Warp = qo head within the kv group.
template <int kH, int kHQ>
__global__ void __launch_bounds__((kHQ / kH) * 32, 2)
sm70_nc_fp16_kernel(
    const __half* __restrict__ q,   // [R, q_rows, kHQ, kD]
    const __half* __restrict__ kv,  // [num_blocks, kH, bs, 2*kD] (K/V last dim)
    const int* __restrict__ bt,     // [R, bt_cols]
    const int* __restrict__ kv_lens,  // [R]
    __half* __restrict__ out,       // [R, q_rows, kHQ, kD]
    float sm_scale,
    int q_rows, int bt_cols, int page_size,
    int64_t kv_s0, int64_t kv_sh, int64_t kv_soff) {
  constexpr int kRatio = kHQ / kH;
  const int r = blockIdx.y;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int head_i = warp;               // 0..kRatio-1
  const int hq = h * kRatio + head_i;     // global qo head

  __shared__ __half k_smem[kN * kD];
  __shared__ __half v_smem[kN * kD];
  __shared__ float s_sh[kRatio][kMaxQ][kN];  // per-warp scaled scores

  const int kv_len = __ldg(kv_lens + r);
  if (kv_len <= 0) return;

  // Load this warp's q rows (zero beyond q_rows so inactive rows are inert).
  float qf[kMaxQ][kLaneDims];
#pragma unroll
  for (int qq = 0; qq < kMaxQ; qq++) {
#pragma unroll
    for (int j = 0; j < kLaneDims; j++) {
      float v = 0.0f;
      if (qq < q_rows)
        v = __half2float(
            q[((int64_t)(r * q_rows + qq) * kHQ + hq) * kD +
              lane * kLaneDims + j]);
      qf[qq][j] = v;
    }
  }

  // Online-softmax state per q row (uniform across the warp — every lane
  // reads the shared scores, so m/l/alpha update identically on all lanes).
  float acc[kMaxQ][kLaneDims];
  float m[kMaxQ], l[kMaxQ];
#pragma unroll
  for (int qq = 0; qq < kMaxQ; qq++) {
    m[qq] = -INFINITY;
    l[qq] = 0.0f;
#pragma unroll
    for (int j = 0; j < kLaneDims; j++) acc[qq][j] = 0.0f;
  }

  const int nchunks = (kv_len + kN - 1) / kN;
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * kN;
    const int n_valid = min(kN, kv_len - t0);

    // Stage K and V for this chunk: each warp stages one token (32 lanes x
    // 4 dims); warps stride through the chunk's tokens.
    const int nwarps = blockDim.x >> 5;
    for (int t = warp; t < n_valid; t += nwarps) {
      const int tok = t0 + t;
      const int page = __ldg(bt + r * bt_cols + tok / page_size);
      const int off = tok % page_size;
      const int64_t base =
          (int64_t)page * kv_s0 + (int64_t)h * kv_sh +
          (int64_t)off * kv_soff;
      const __half* ksrc = kv + base;
      const __half* vsrc = kv + base + kD;
#pragma unroll
      for (int j = 0; j < kLaneDims; j++) {
        const int d = lane * kLaneDims + j;
        k_smem[t * kD + d] = ksrc[d];
        v_smem[t * kD + d] = vsrc[d];
      }
    }
    __syncthreads();

    // Phase 1: scaled scores for every active q row into shared memory.
    for (int t = 0; t < n_valid; t++) {
#pragma unroll
      for (int qq = 0; qq < q_rows; qq++) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < kLaneDims; j++)
          s = fmaf(__half2float(k_smem[t * kD + lane * kLaneDims + j]),
                   qf[qq][j], s);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          s += __shfl_xor_sync(0xFFFFFFFF, s, o);
        if (lane == 0) s_sh[warp][qq][t] = s * sm_scale;
      }
    }
    __syncwarp();

    // Running max over the chunk's scaled scores (uniform across lanes).
    float alpha[kMaxQ];
#pragma unroll
    for (int qq = 0; qq < q_rows; qq++) {
      float row_max = -INFINITY;
      for (int t = 0; t < n_valid; t++)
        row_max = fmaxf(row_max, s_sh[warp][qq][t]);
      const float m_new = fmaxf(m[qq], row_max);
      alpha[qq] = (m[qq] == -INFINITY) ? 0.0f : expf(m[qq] - m_new);
      l[qq] *= alpha[qq];
      m[qq] = m_new;
    }

    // Rescale the running accumulator, then accumulate p @ V.
#pragma unroll
    for (int qq = 0; qq < q_rows; qq++) {
#pragma unroll
      for (int j = 0; j < kLaneDims; j++)
        acc[qq][j] *= alpha[qq];
    }
    for (int t = 0; t < n_valid; t++) {
#pragma unroll
      for (int qq = 0; qq < q_rows; qq++) {
        const float p = expf(s_sh[warp][qq][t] - m[qq]);
        l[qq] += p;
#pragma unroll
        for (int j = 0; j < kLaneDims; j++)
          acc[qq][j] =
              fmaf(p, __half2float(v_smem[t * kD + lane * kLaneDims + j]),
                   acc[qq][j]);
      }
    }
    __syncthreads();
  }

  // Normalize and write out[r, qq, hq, d] for this qo head.
  for (int qq = 0; qq < q_rows; qq++) {
    const float inv_l = (l[qq] > 0.0f) ? 1.0f / l[qq] : 0.0f;
    __half* orow = out + ((int64_t)(r * q_rows + qq) * kHQ + hq) * kD;
#pragma unroll
    for (int j = 0; j < kLaneDims; j++)
      orow[lane * kLaneDims + j] =
          __float2half(acc[qq][j] * inv_l);
  }
}

}  // namespace

void sm70_nc_fp16(torch::stable::Tensor& q, torch::stable::Tensor& kv,
                  torch::stable::Tensor& block_table,
                  torch::stable::Tensor& kv_lens, torch::stable::Tensor& out,
                  double sm_scale, int64_t q_rows) {
  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  STD_TORCH_CHECK(q.size(1) == q_rows && q_rows >= 1 && q_rows <= kMaxQ,
                  "sm70_nc_fp16: q dim 1 must equal q_rows (1..8)");
  STD_TORCH_CHECK(q.size(3) == kD, "sm70_nc_fp16: q head_dim must be 128");
  const int64_t HQ = q.size(2);
  const int64_t H = kv.size(1);
  const int64_t page_size = kv.size(2);
  STD_TORCH_CHECK(H == 2 && HQ == 8,
                  "sm70_nc_fp16: only the (2 kv, 8 qo) TP4 geometry is built");
  STD_TORCH_CHECK(kv.size(3) == 2 * kD,
                  "sm70_nc_fp16: kv last dim must be 2*head_dim");
  STD_TORCH_CHECK(kv.stride(3) == 1,
                  "sm70_nc_fp16: kv last dim must be contiguous");
  const int64_t R = q.size(0);
  const int64_t bt_cols = block_table.size(1);
  STD_TORCH_CHECK(block_table.size(0) == R && kv_lens.size(0) == R,
                  "sm70_nc_fp16: table/len mismatch");
  STD_TORCH_CHECK(out.size(0) == R && out.size(1) == q_rows &&
                      out.size(2) == HQ && out.size(3) == kD,
                  "sm70_nc_fp16: out shape mismatch");

  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());
  dim3 grid(H, R);
  dim3 block((HQ / H) * 32);
  sm70_nc_fp16_kernel<2, 8><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const __half*>(kv.data_ptr()),
      reinterpret_cast<const int*>(block_table.data_ptr()),
      reinterpret_cast<const int*>(kv_lens.data_ptr()),
      reinterpret_cast<__half*>(out.mutable_data_ptr()),
      static_cast<float>(sm_scale),
      static_cast<int>(q_rows), static_cast<int>(bt_cols),
      static_cast<int>(page_size), kv.stride(0), kv.stride(1),
      kv.stride(2));
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "sm70_nc_fp16 launch failed: ",
                  cudaGetErrorString(err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_nc_fp16_ops) {
  sm70_nc_fp16_ops.def(
      "sm70_nc_fp16(Tensor q, Tensor kv_cache, Tensor block_table, "
      "Tensor kv_lens, Tensor out, float sm_scale, int q_rows) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_nc_fp16_ops) {
  sm70_nc_fp16_ops.impl("sm70_nc_fp16", TORCH_BOX(&::sm70_nc_fp16));
}
