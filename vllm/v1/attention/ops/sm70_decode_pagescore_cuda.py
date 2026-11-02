"""CUDA page scorer for the SM70 nvfp4 decode cache (P3, engine-usable).

Scores every 16-token KV page against the current query without dequantizing the
whole cache: each page samples `S` token rows and, for every q head of that kv
head, keeps `max(sum_d q[head, d] * K[page, token, d])`.

Layout matches the decode kernels: page bytes are flat
`[K data H*N*DD][K scales H*N*SD][V data][V scales]`, `N=16`, `DD=D/2`,
`SD=D/16`. Only K is read, so one pass costs
`pages * S * G * D * 0.5 B` per kv head (score only; the attention itself then
runs on the selected pages).
"""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float e2m1_score(int n) {
  const float L[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = L[n & 7];
  return (n & 8) ? -v : v;
}

__device__ __forceinline__ float e4m3_score(unsigned char b) {
  int s = (b & 0x80) ? -1 : 1;
  int e = (b >> 3) & 0xF, m = b & 7;
  float v = (e == 0) ? (m * 0.001953125f) : ((1.f + m * 0.125f) * exp2f((float)(e - 7)));
  return s * v;
}

// grid (H, ntiles), block 32. Lane owns dims [lane*8, lane*8+8) = 4 packed bytes.
__global__ void page_scores_kernel(
    const __half* __restrict__ q,         // [HQ, D]
    const unsigned char* __restrict__ kv, // [B, ...] flat pages
    const int* __restrict__ bt,           // [bt_cols]
    const int* __restrict__ kvlens,       // [R]
    float* __restrict__ out,              // [H, ntiles]
    int HQ, int D, int NP, int DD, int SD, int page_bytes,
    int bt_cols, int ntiles, int samples, int r, int G) {
  const int h = blockIdx.x;
  const int pg = blockIdx.y;
  const int lane = threadIdx.x;
  const int kv_len = kvlens[r];
  float best = -1e30f;
  if (pg * NP < kv_len) {
    const int bid = bt[r * bt_cols + pg];
    const unsigned char* base = kv + (long)bid * page_bytes;
    const unsigned char* kdata = base + (long)h * (NP * DD);
    const unsigned char* kscale =
        base + (long)gridDim.x * (NP * DD) + (long)h * (NP * SD);
    for (int s = 0; s < samples; ++s) {
      const int tok = (int)((long)s * NP / samples);
      if (pg * NP + tok >= kv_len) break;
      const unsigned char* kd = kdata + (long)tok * DD + lane * 4;
      const float sc = e4m3_score(kscale[(long)tok * SD + (lane >> 1)]);
      const unsigned int w = *reinterpret_cast<const unsigned int*>(kd);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int qh = h * G + g;
        if (qh >= HQ) break;
        const __half* qp = q + (long)qh * D + lane * 8;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned char byte = (w >> (j * 8)) & 0xFFu;
          acc += __half2float(qp[j * 2]) * e2m1_score(byte & 0xF) * sc;
          acc += __half2float(qp[j * 2 + 1]) * e2m1_score((byte >> 4) & 0xF) * sc;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_xor_sync(0xffffffffu, acc, off);
        best = fmaxf(best, acc);
      }
    }
  }
  if (lane == 0) out[h * ntiles + pg] = best;
}

void page_scores(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
                 torch::Tensor kvlens, torch::Tensor out, int64_t samples,
                 int64_t r) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = out.size(0), ntiles = out.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  dim3 grid(H, ntiles);
  page_scores_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      out.data_ptr<float>(), HQ, D, NP, DD, SD, page_bytes,
      (int)bt.size(1), ntiles, (int)samples, (int)r, G);
}


// Coalesced page-mean K: grid (H, ceil(pages/PP)), block 256 (one thread per dim).
// Reads each page's K rows sequentially (128 B/token, coalesced) and writes the
// page mean, so scoring becomes one small GEMV per head instead of a gather per
// page. PP pages per CTA keeps the launch count small.
__global__ void page_mean_k_kernel(
    const unsigned char* __restrict__ kv, const int* __restrict__ bt,
    const int* __restrict__ kvlens, __half* __restrict__ kmean,
    int D, int NP, int DD, int SD, int page_bytes, int bt_cols, int ntiles,
    int r, int PP) {
  const int h = blockIdx.x;
  const int p0 = blockIdx.y * PP;
  const int d = threadIdx.x;               // 0..255, one dim
  const int sub = d >> 4;                  // 16-value scale group
  const int kv_len = kvlens[r];
  for (int pg = p0; pg < p0 + PP && pg < ntiles; ++pg) {
    if (pg * NP >= kv_len) break;
    const int bid = bt[r * bt_cols + pg];
    const unsigned char* kdata = kv + (long)bid * page_bytes + (long)h * (NP * DD);
    const unsigned char* kscale = kv + (long)bid * page_bytes +
                                  (long)gridDim.x * (NP * DD) +
                                  (long)h * (NP * SD);
    float acc = 0.f;
    int n = 0;
    for (int t = 0; t < NP; ++t) {
      if (pg * NP + t >= kv_len) break;
      const unsigned char byte = kdata[(long)t * DD + (d >> 1)];
      const int code = ((d & 1) ? (byte >> 4) : (byte & 0xF));
      const float sc = e4m3_score(kscale[(long)t * SD + sub]);
      acc += e2m1_score(code) * sc;
      ++n;
    }
    kmean[((long)h * ntiles + pg) * D + d] = __float2half(acc / (float)(n ? n : 1));
  }
}

void page_mean_k(torch::Tensor kv, torch::Tensor bt, torch::Tensor kvlens,
                 torch::Tensor kmean, int64_t r, int64_t pp) {
  const int H = kmean.size(0), ntiles = kmean.size(1), D = kmean.size(2);
  const int NP = kv.size(2), F = kv.size(3);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  dim3 grid(H, (ntiles + (int)pp - 1) / (int)pp);
  page_mean_k_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      reinterpret_cast<__half*>(kmean.data_ptr<at::Half>()), D, NP, DD, SD,
      page_bytes, (int)bt.size(1), ntiles, (int)r, (int)pp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("page_scores", &page_scores);
  m.def("page_mean_k", &page_mean_k);
}
"""

_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(name="sm70_decode_pagescore", cpp_sources="",
                           cuda_sources=_CUDA, verbose=False,
                           extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _MOD


def page_scores(q, kv, bt, kvlens, out, samples=4, r=0):
    _mod().page_scores(q, kv, bt, kvlens, out, samples, r)
    return out


def page_mean_k(kv, bt, kvlens, kmean, r=0, pp=16):
    """Page-mean K, coalesced: [H, pages, D] fp16 in `kmean`."""
    _mod().page_mean_k(kv, bt, kvlens, kmean, r, pp)
    return kmean
