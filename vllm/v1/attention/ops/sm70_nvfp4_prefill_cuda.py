# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project

"""Hand-written SM70 (V100) NVFP4 paged PREFILL attention (WMMA tensor cores).

Drop-in faster alternative to the Triton ``sm70_nvfp4_paged_prefill``: same
signature, same masking semantics (query row ``j`` sits at
``pos = kv_len - q_len + q0 + j`` and attends to
``[max(0, pos - window_left), pos]``), same LBHNC NVFP4 page layout.

Why it is faster than both available paths (measured at 65K, 10 layers,
TP1 V100, ncu + nsys):
  * flashinfer ``BatchPrefillWithPagedKVCacheKernel``: 56.9 s, 6.2 TFLOPS
    (~5% of the 125 TFLOPS fp16 peak; 6.25% warps active, 80 KB smem ->
    1 CTA/SM).
  * Triton native prefill: no tensor cores on sm_70 (``tl.dot`` is lowered to
    scalar FMA), slower than flashinfer.
  * this kernel: 16.1 s, 21.8 TFLOPS (3.5x flashinfer).

Design (vs the earlier BM=16 prototype, which ncu showed at 6.4% tensor pipe,
L1-bound with 625M shared-memory bank conflicts):
  * BM=64 q rows per CTA: K/V dequant is amortized over 4x more rows.
  * Vectorized dequant: each thread turns one 16-byte NVFP4 group into four
    ``uint4`` shared-memory stores instead of 32 scalar 2-byte stores.
  * O in registers (warp-private accumulators), so the online-softmax rescale
    costs no shared memory at all. This needs the real Volta accumulator
    layout, which is NOT ``lane/4``; dumps showed
    ``row = (lane&1) + 4*((lane>>4)&1) + 8*((lane>>2)&1)`` (+2 when ``i&2``).
  * 16 warps / CTA, 1 CTA/SM, ~95 KB dynamic smem, 4 barriers per KV tile.
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
using namespace nvcuda;

#define HD 256
#define BM 64
#define BN 64
#define QS 264
#define KVS 264
#define SS 72
#define THREADS 512

__device__ __forceinline__ float e2m1(int n) {
  const float L[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = L[n & 7];
  return (n & 8) ? -v : v;
}
__device__ __forceinline__ float e4m3(unsigned char b) {
  int s = (b & 0x80) ? -1 : 1, e = (b >> 3) & 0xF, m = b & 7;
  if (e == 15 && m == 7) return 448.f;
  float v = (e == 0) ? (m * 0.001953125f)
                     : ((1.f + m * 0.125f) * exp2f((float)(e - 7)));
  return s * v;
}

// One NVFP4 group of 32 dims: 16 data bytes + 2 fp8 scales -> 32 halves.
__device__ __forceinline__ void deq_unit(__half* sKV, const unsigned char* pd,
                                         const long data_off, const long sf_off,
                                         int row, int sub, float scale) {
  const unsigned char* bs = pd + sf_off;
  float s0 = e4m3(bs[0]) * scale, s1 = e4m3(bs[1]) * scale;
  uint4 v = *reinterpret_cast<const uint4*>(pd + data_off);
  unsigned int w4[4] = {v.x, v.y, v.z, v.w};
  int d0 = sub * 32;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    unsigned int by = w4[w];
    float sf = (w < 2) ? s0 : s1;
    union { __half2 h2[4]; uint4 u; } pk;
#pragma unroll
    for (int bb = 0; bb < 4; bb++) {
      unsigned char b = (by >> (bb * 8)) & 0xFF;
      pk.h2[bb] = __floats2half2_rn(e2m1(b & 0xF) * sf,
                                    e2m1((b >> 4) & 0xF) * sf);
    }
    *reinterpret_cast<uint4*>(&sKV[row * KVS + d0 + w * 8]) = pk.u;
  }
}

// grid (R, HQ, q_tiles); block THREADS
__global__ __launch_bounds__(THREADS, 1)
void fa_pf(const __half* __restrict__ Q, const unsigned char* __restrict__ kv,
           __half* __restrict__ Out, const int* __restrict__ bt,
           const int* __restrict__ qo_indptr, const int* __restrict__ kvlens,
           float k_scale, float v_scale, float sm_scale, long page,
           long side_off, int bt_cols, int window, int H, int HQ, int NP,
           int DD, int SD, int F) {
  extern __shared__ char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sKV = sQ + BM * QS;
  float* sS = reinterpret_cast<float*>(sKV + BN * KVS);
  __half* sP = reinterpret_cast<__half*>(sS + BM * SS);
  float* sMax = reinterpret_cast<float*>(sP + BM * SS);
  float* sSum = sMax + BM;
  float* sExp = sSum + BM;

  const int r = blockIdx.x, qh = blockIdx.y, qt = blockIdx.z;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int qo0 = qo_indptr[r];
  const int q_len = qo_indptr[r + 1] - qo0;
  const int q0 = qt * BM;
  if (q0 >= q_len) return;
  const int valid_q = min(BM, q_len - q0);
  const int kv_len = kvlens[r];
  const int kh = qh / (HQ / H);
  const int pos0 = kv_len - q_len + q0;   // absolute position of row 0

  for (int i = tid; i < BM * (HD / 8); i += THREADS) {
    int row = i / (HD / 8), c8 = (i % (HD / 8)) * 8;
    if (row < valid_q) {
      long off = ((long)(qo0 + q0 + row) * HQ + qh) * HD + c8;
      *reinterpret_cast<uint4*>(&sQ[row * QS + c8]) =
          *reinterpret_cast<const uint4*>(Q + off);
    } else {
      *reinterpret_cast<uint4*>(&sQ[row * QS + c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  if (tid < BM) { sMax[tid] = -1e30f; sSum[tid] = 0.f; }

  const int mt = warp & 3, ntqk = warp >> 2;
  const int nbase = 4 * ntqk;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[4];
#pragma unroll
  for (int t = 0; t < 4; t++) wmma::fill_fragment(o[t], 0.f);
  const int rbase = (lane & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);
  const int orow = mt * 16 + rbase;
  __syncthreads();

  const int maxpos = pos0 + valid_q - 1;
  const int hi = min(kv_len, maxpos + 1);
  const int nkv = (hi + BN - 1) / BN;
  int kt_begin = 0;
  if (window >= 0) {
    int lo = pos0 - window;
    if (lo > 0) kt_begin = lo / BN;
  }
  const long kdata0 = (long)kh * (NP * DD);
  const long ksf0 = (long)H * NP * DD + (long)kh * (NP * SD);
  const long vdata0 = side_off + kdata0;
  const long vsf0 = (long)H * NP * DD + side_off + (long)kh * (NP * SD);

  for (int kt = kt_begin; kt < nkv; kt++) {
    const int kstart = kt * BN, valid_k = min(BN, kv_len - kstart);
    {   // ---- K dequant ----
      int row = tid / 8, sub = tid % 8;
      if (row < valid_k) {
        int tok = kstart + row, b = tok / NP, nn = tok % NP;
        const unsigned char* pd = kv + (long)bt[r * bt_cols + b] * page;
        deq_unit(sKV, pd, kdata0 + (long)nn * DD + sub * 16,
                 ksf0 + (long)nn * SD + sub * 2, row, sub, k_scale);
      } else {
        int d0 = sub * 32;
#pragma unroll
        for (int d = 0; d < 32; d++) sKV[row * KVS + d0 + d] = __float2half(0.f);
      }
    }
    __syncthreads();
    {   // ---- QK: warp -> (m-tile, n-tile) ----
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> st;
      wmma::fill_fragment(st, 0.f);
#pragma unroll
      for (int k = 0; k < HD / 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
        wmma::load_matrix_sync(qf, &sQ[mt * 16 * QS + k * 16], QS);
        wmma::load_matrix_sync(kf, &sKV[ntqk * 16 * KVS + k * 16], KVS);
        wmma::mma_sync(st, qf, kf, st);
      }
      wmma::store_matrix_sync(&sS[mt * 16 * SS + ntqk * 16], st, SS,
                              wmma::mem_row_major);
    }
    __syncthreads();
    {   // ---- V dequant (sKV is free again) ----
      int row = tid / 8, sub = tid % 8;
      if (row < valid_k) {
        int tok = kstart + row, b = tok / NP, nn = tok % NP;
        const unsigned char* pd = kv + (long)bt[r * bt_cols + b] * page;
        deq_unit(sKV, pd, vdata0 + (long)nn * DD + sub * 16,
                 vsf0 + (long)nn * SD + sub * 2, row, sub, v_scale);
      } else {
        int d0 = sub * 32;
#pragma unroll
        for (int d = 0; d < 32; d++) sKV[row * KVS + d0 + d] = __float2half(0.f);
      }
    }
    {   // ---- softmax (8 lanes per row, 8 columns each) ----
      int row = tid / 8, cg = tid % 8, qabs = pos0 + row;
      int lo = (window >= 0) ? max(0, qabs - window) : 0;
      float tmp[8], lm = -1e30f;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int col = cg * 8 + j, tok = kstart + col;
        float sc = (row < valid_q && col < valid_k && tok <= qabs && tok >= lo)
                       ? sS[row * SS + col] * sm_scale : -1e30f;
        tmp[j] = sc; lm = fmaxf(lm, sc);
      }
      lm = fmaxf(lm, __shfl_xor_sync(0xffffffff, lm, 1));
      lm = fmaxf(lm, __shfl_xor_sync(0xffffffff, lm, 2));
      lm = fmaxf(lm, __shfl_xor_sync(0xffffffff, lm, 4));
      float oldm = sMax[row], newm = fmaxf(oldm, lm), ed = expf(oldm - newm);
      float sd = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++) { tmp[j] = expf(tmp[j] - newm); sd += tmp[j]; }
      sd += __shfl_xor_sync(0xffffffff, sd, 1);
      sd += __shfl_xor_sync(0xffffffff, sd, 2);
      sd += __shfl_xor_sync(0xffffffff, sd, 4);
      if (cg == 0) {
        sMax[row] = newm;
        sSum[row] = sSum[row] * ed + sd;
        sExp[row] = ed;
      }
#pragma unroll
      for (int j = 0; j < 8; j++)
        sP[row * SS + cg * 8 + j] = __float2half(tmp[j]);
    }
    __syncthreads();
    {   // ---- O rescale in registers ----
      float e0 = sExp[orow], e2 = sExp[orow + 2];
#pragma unroll
      for (int t = 0; t < 4; t++)
#pragma unroll
        for (int i = 0; i < 8; i++) o[t].x[i] *= ((i & 2) ? e2 : e0);
    }
    {   // ---- PV ----
#pragma unroll
      for (int t = 0; t < 4; t++) {
        int nt = nbase + t;
#pragma unroll
        for (int k = 0; k < BN / 16; k++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
          wmma::load_matrix_sync(pf, &sP[mt * 16 * SS + k * 16], SS);
          wmma::load_matrix_sync(vf, &sKV[k * 16 * KVS + nt * 16], KVS);
          wmma::mma_sync(o[t], pf, vf, o[t]);
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  __half* stage = sQ;   // Q is dead now
  {
    float inv0 = 1.f / sSum[orow], inv2 = 1.f / sSum[orow + 2];
    int cpre = 2 * ((lane >> 1) & 1) + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int t = 0; t < 4; t++) {
      int cbase = (nbase + t) * 16;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int c = cbase + cpre + (i & 1) + 4 * ((i >> 2) & 1);
        int rr = orow + ((i & 2) ? 2 : 0);
        stage[rr * QS + c] = __float2half(o[t].x[i] * ((i & 2) ? inv2 : inv0));
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < valid_q * (HD / 8); i += THREADS) {
    int row = i / (HD / 8), c8 = (i % (HD / 8)) * 8;
    long off = ((long)(qo0 + q0 + row) * HQ + qh) * HD + c8;
    *reinterpret_cast<uint4*>(Out + off) =
        *reinterpret_cast<const uint4*>(&stage[row * QS + c8]);
  }
}

static bool g_attr = false;
void run(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
         torch::Tensor qo_indptr, torch::Tensor kvlens, torch::Tensor out,
         double k_scale, double v_scale, double sm_scale, int64_t page,
         int64_t side_off, int64_t bt_cols, int64_t window,
         int64_t max_q_len) {
  const int R = bt.size(0), HQ = q.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9;          // 16  = D/16  (F = D/2 + D/16)
  const int DD = F - SD;         // 128 = D/2
  const int H = kv.size(1) / 2;
  int smem = (BM * QS + BN * KVS) * 2 + BM * SS * 4 + BM * SS * 2 + 3 * BM * 4;
  if (!g_attr) {
    cudaFuncSetAttribute(fa_pf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g_attr = true;
  }
  int qtiles = max(1, (int)((max_q_len + BM - 1) / BM));
  dim3 g(R, HQ, qtiles);
  fa_pf<<<g, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), bt.data_ptr<int>(),
      qo_indptr.data_ptr<int>(), kvlens.data_ptr<int>(), (float)k_scale,
      (float)v_scale, (float)sm_scale, (long)page, (long)side_off,
      (int)bt_cols, (int)window, H, HQ, NP, DD, SD, F);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
"""

_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(
            name="sm70_nvfp4_prefill_cuda_v1",
            cpp_sources="",
            cuda_sources=_CUDA,
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _MOD


def warmup_compile(device=None):
    """Compile ahead of the first forward (keeps the first prefill fast)."""
    _mod()


def try_paged_prefill_cuda(q, kv_cache, block_table, qo_indptr, kv_lens, out,
                           k_scale, v_scale, sm_scale, window_left,
                           max_q_len) -> bool:
    """Run the hand-written kernel; return False to fall back to Triton."""
    if os.environ.get("VLLM_SM70_PREFILL_CUDA") != "1":
        return False
    if not (q.is_cuda and q.dtype == torch.float16
            and q.is_contiguous() and out.is_contiguous()):
        return False
    D = q.shape[2]
    H = kv_cache.shape[1] // 2
    if D != 256 or q.shape[1] != 16 or H != 2:
        return False
    if max_q_len <= 0 or block_table.shape[0] != qo_indptr.shape[0] - 1:
        return False
    _mod().run(q, kv_cache, block_table, qo_indptr, kv_lens, out,
               k_scale, v_scale, sm_scale, kv_cache.stride(0),
               H * kv_cache.shape[2] * kv_cache.shape[3],
               block_table.shape[1], window_left, max_q_len)
    return True
