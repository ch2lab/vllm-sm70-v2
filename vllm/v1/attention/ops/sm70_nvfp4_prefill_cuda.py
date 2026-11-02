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
  * this kernel: 16.1 s, 21.8 TFLOPS (3.5x flashinfer); 14.6 s with the fp4
    lookup table below (single-layer 65K q=4096: 197 -> 179 ms, bit-identical).

Design (vs the earlier BM=16 prototype, which ncu showed at 6.4% tensor pipe,
L1-bound with 625M shared-memory bank conflicts):
  * BM=64 q rows per CTA: K/V dequant is amortized over 4x more rows.
  * Vectorized dequant: each thread turns one 16-byte NVFP4 group into four
    ``uint4`` shared-memory stores instead of 32 scalar 2-byte stores.
  * fp4 -> float2 table in shared memory (2 KB, 98048 B total): one 8-byte load
    per packed byte replaces two dynamic-index selects over ``L[8]``, worth
    1.13x on the whole kernel and bit-identical. The remaining 46% of tile time
    is the dequant's smem stores plus the 2x-amortisable part of the LUT, and the
    mma phases sit at ~1/3 of peak because of shared-memory fragment access
    patterns (swizzled/register-blocked tiling is the next step, not a pad).
  * O in registers (warp-private accumulators), so the online-softmax rescale
    costs no shared memory at all. This needs the real Volta accumulator
    layout, which is NOT ``lane/4``; dumps showed
    ``row = (lane&1) + 4*((lane>>4)&1) + 8*((lane>>2)&1)`` (+2 when ``i&2``).
  * 16 warps / CTA, 1 CTA/SM, 97.75 KB dynamic smem, 4 barriers per KV tile.
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
using namespace nvcuda;

#define HD 256
#define BM 64
#define BN 64
#define QS 264
#define KVS 272
#define SS 72            // sP stride (fp16: multiple of 8 halves)
#define SSS 68           // sS stride (fp32: 4-byte aligned is enough)
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
// ``lut`` holds all 256 packed-byte -> (lo, hi) nibble pairs, so each byte costs
// one shared load instead of two table selects.
__device__ __forceinline__ void deq_unit(__half* sKV, const unsigned char* pd,
                                         const long data_off, const long sf_off,
                                         int row, int sub, float scale,
                                         const float2* __restrict__ lut) {
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
      float2 f = lut[b];
      pk.h2[bb] = __floats2half2_rn(f.x * sf, f.y * sf);
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
           long side_off, int bt_cols, int window, int sink,
           const unsigned int* __restrict__ keep, int keep_words, int H, int HQ,
           int NP,
           int DD, int SD, int F) {
  extern __shared__ char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sKV = sQ + BM * QS;
  float* sS = reinterpret_cast<float*>(sKV + BN * KVS);
  __half* sP = reinterpret_cast<__half*>(sS + BM * SSS);
  float* sMax = reinterpret_cast<float*>(sP + BM * SS);
  float* sSum = sMax + BM;
  float* sExp = sSum + BM;
  // fp4 -> float2 table, one 8-byte load per packed byte instead of two
  // dynamic-index selects (measured 1.13x on the whole kernel).
  float2* sLut = reinterpret_cast<float2*>(sExp + BM);

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
  for (int i = tid; i < 256; i += THREADS)
    sLut[i] = make_float2(e2m1(i & 15), e2m1(i >> 4));

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
  // Bounded reading (Flash-Next style): keep the first ``sink`` tokens and the
  // last ``window`` tokens, skip the KV tiles in between.  ``window < 0`` and
  // ``sink == 0`` reproduce the plain dense loop exactly.
  int kt_begin = 0, kt_sink = 0;
  if (window >= 0) {
    int lo = pos0 - window;
    if (lo > 0) {
      kt_begin = lo / BN;
      kt_sink = min(nkv, (sink + BN - 1) / BN);
    }
  }
  const long kdata0 = (long)kh * (NP * DD);
  const long ksf0 = (long)H * NP * DD + (long)kh * (NP * SD);
  const long vdata0 = side_off + kdata0;
  const long vsf0 = (long)H * NP * DD + side_off + (long)kh * (NP * SD);

  for (int kt = 0; kt < nkv; kt++) {
    if (kt >= kt_sink && kt < kt_begin) continue;
    if (keep != nullptr &&
        !(keep[((long)r * HQ + qh) * keep_words + (kt >> 5)] & (1u << (kt & 31))))
      continue;
    const int kstart = kt * BN, valid_k = min(BN, kv_len - kstart);
    {   // ---- K dequant ----
      int row = tid / 8, sub = tid % 8;
      if (row < valid_k) {
        int tok = kstart + row, b = tok / NP, nn = tok % NP;
        const unsigned char* pd = kv + (long)bt[r * bt_cols + b] * page;
        deq_unit(sKV, pd, kdata0 + (long)nn * DD + sub * 16,
                 ksf0 + (long)nn * SD + sub * 2, row, sub, k_scale, sLut);
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
      wmma::store_matrix_sync(&sS[mt * 16 * SSS + ntqk * 16], st, SSS,
                              wmma::mem_row_major);
    }
    __syncthreads();
    {   // ---- V dequant (sKV is free again) ----
      int row = tid / 8, sub = tid % 8;
      if (row < valid_k) {
        int tok = kstart + row, b = tok / NP, nn = tok % NP;
        const unsigned char* pd = kv + (long)bt[r * bt_cols + b] * page;
        deq_unit(sKV, pd, vdata0 + (long)nn * DD + sub * 16,
                 vsf0 + (long)nn * SD + sub * 2, row, sub, v_scale, sLut);
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
        float sc = (row < valid_q && col < valid_k && tok <= qabs
                    && (tok >= lo || tok < sink))
                       ? sS[row * SSS + col] * sm_scale : -1e30f;
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


// ---- training-free block selector for sparse prefill (env-gated, default off)
// Stage 1: one warp per 64-key tile scores the tile with max(q.k) over 4 sampled
// keys x 4 sampled queries (2 proxy q heads per kv head).  Stage 2 turns the
// scores into a per-tile keep bitmap (sink + recent tiles forced).
// Purely subtractive in fa_pf: kept tiles are bit-identical to dense.
__global__ __launch_bounds__(32, 8)
void sel_scores(const __half* __restrict__ Q, const unsigned char* __restrict__ kv,
                const int* __restrict__ bt, const int* __restrict__ qo_indptr,
                const int* __restrict__ kvlens, float* __restrict__ scores,
                float* __restrict__ radius,
                float k_scale, float sm_scale, long page, int bt_cols, int H,
                int HQ, int NP, int DD, int SD, int ntiles) {
  const int r = blockIdx.x, kh = blockIdx.y, t = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int qo0 = qo_indptr[r];
  const int q_len = qo_indptr[r + 1] - qo0;
  const int kv_len = kvlens[r];
  const int base = t * BN;
  float best = -1e30f;
  if (base < kv_len && q_len > 0) {
    const int G = HQ / H;
    const int half = lane >> 4;            // this half-warp scores queries half, half+2
    const int l = lane & 15;               // 16 dims per lane
    // Block score = log-sum-exp of the sampled logits (BFLA-style mass proxy),
    // maxed over the sampled queries.  Sampling 8 keys inside the 64-key tile.
    float mx[2] = {-1e30f, -1e30f}, sm[2] = {0.f, 0.f};
    float ksum = 0.f, ksq = 0.f;   // per-lane key moments -> block radius proxy
    for (int ki = 0; ki < 8; ki++) {
      const int tok = min(base + ki * 8 + 4, kv_len - 1);
      const long pg = (long)bt[r * bt_cols + (tok >> 4)] * page;
      const int nn = tok & 15;
      const unsigned char* kd = kv + pg + (long)kh * (NP * DD) + nn * DD + l * 8;
      const unsigned char* ks = kv + pg + (long)H * NP * DD + (long)kh * (NP * SD)
                                + nn * SD + l;
      uint2 v = *reinterpret_cast<const uint2*>(kd);
      const float sf = e4m3(ks[0]) * k_scale;
      float kd16[16];
      (void)0;
      const unsigned int w2[2] = {v.x, v.y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
        unsigned int by = w2[w];
#pragma unroll
        for (int bb = 0; bb < 4; bb++) {
          unsigned char b = (by >> (bb * 8)) & 0xFF;
          kd16[w * 8 + bb * 2 + 0] = e2m1(b & 0xF) * sf;
          kd16[w * 8 + bb * 2 + 1] = e2m1((b >> 4) & 0xF) * sf;
        }
      }
      if (lane < 16) {
#pragma unroll
        for (int d = 0; d < 16; d++) { ksum += kd16[d]; ksq += kd16[d] * kd16[d]; }
      }
#pragma unroll
      for (int qq = 0; qq < 2; qq++) {
        const int qi = half + 2 * qq;
        const int qs = qi == 0 ? 0 : qi == 1 ? q_len / 3
                       : qi == 2 ? (2 * q_len) / 3 : q_len - 1;
        const __half* qp = Q + ((long)(qo0 + qs) * HQ + kh * G) * HD + l * 16;
        float acc = 0.f;
#pragma unroll
        for (int d = 0; d < 16; d++) acc += __half2float(qp[d]) * kd16[d];
#pragma unroll
        for (int o = 8; o; o >>= 1)
          acc += __shfl_xor_sync(0xffffffffu, acc, o, 16);
        const float sc = acc * sm_scale;
        const float m2 = fmaxf(mx[qq], sc);
        sm[qq] = sm[qq] * __expf(mx[qq] - m2) + __expf(sc - m2);
        mx[qq] = m2;
      }
    }
    float lse = -1e30f;
#pragma unroll
    for (int qq = 0; qq < 2; qq++) lse = fmaxf(lse, mx[qq] + __logf(sm[qq]));
    best = fmaxf(lse, __shfl_xor_sync(0xffffffffu, lse, 16));
    // radius proxy: std of the sampled keys (mean dilution risk signal)
    float s1 = ksum, s2 = ksq;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) {
      s1 += __shfl_down_sync(0xffffffffu, s1, o, 16);
      s2 += __shfl_down_sync(0xffffffffu, s2, o, 16);
    }
    const float ns = 8.f * 16.f;              // 8 samples x 16 dims per lane group
    const float mean_k = s1 / ns;
    const float var_k = fmaxf(s2 / ns - mean_k * mean_k, 0.f);
    if (lane == 0)
      radius[((long)blockIdx.x * gridDim.y + kh) * ntiles + t] = __fsqrt_rn(var_k);
  }
  if (lane == 0)
    scores[((long)blockIdx.x * gridDim.y + kh) * ntiles + t] = best;
}

__global__ void sel_keep(const float* __restrict__ scores,
                         const float* __restrict__ radius, int* __restrict__ keep,
                         const int* __restrict__ kvlens, float tau,
                         float tau_rescue, float rescue_k, int sink_tiles,
                         int local_tiles, int ntiles, int words) {
  const int r = blockIdx.x, kh = blockIdx.y;
  const long sb = ((long)r * gridDim.y + kh) * ntiles;
  extern __shared__ float sh[];          // sh[0..127] max, sh[128..255] radius sum
  float* sh2 = sh + blockDim.x;
  float mx = -1e30f, rsum = 0.f;
  for (int t = threadIdx.x; t < ntiles; t += blockDim.x) {
    mx = fmaxf(mx, scores[sb + t]);
    rsum += radius[sb + t];
  }
  sh[threadIdx.x] = mx;
  sh2[threadIdx.x] = rsum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + st]);
      sh2[threadIdx.x] += sh2[threadIdx.x + st];
    }
    __syncthreads();
  }
  const int nkv = (kvlens[r] + BN - 1) / BN;
  // mass rule: keep a block when its share of the softmax mass is >= tau,
  // i.e. score >= max + ln(tau) in logit space.
  const float thr = sh[0] + __logf(tau);
  // RBS-style rescue: blocks with an unusually large spread (mean-dilution risk)
  // are kept at the looser threshold tau_rescue.
  const float thr_r = sh[0] + __logf(tau_rescue);
  const float rad_thr = (sh2[0] / max(nkv, 1)) * rescue_k;
  // the forced recent window scales with the document: clamp(nkv/4, L/4, L)
  const int loc = min(local_tiles, max(local_tiles / 4, nkv / 4));
  for (int wd = threadIdx.x; wd < words; wd += blockDim.x) {
    unsigned int bits = 0;
    for (int b = 0; b < 32; b++) {
      const int t = wd * 32 + b;
      bool kp = false;
      if (t < nkv) {
        const float sc = scores[sb + t];
        kp = (t < sink_tiles) || (t >= nkv - loc) || (sc >= thr) ||
             (radius[sb + t] >= rad_thr && sc >= thr_r);
      }
      if (kp) bits |= 1u << b;
    }
    keep[((long)r * gridDim.y + kh) * words + wd] = (int)bits;
  }
}


// ---- calibration only: per-(q head, tile) block mass ------------------------
// DuoAttention hypothesis test: does a minority of q heads carry the long-range
// mass?  One query row (the chunk's last) x 8 sampled keys per 64-key tile,
// log-sum-exp of the logits -> a per-tile mass for that head.
__global__ __launch_bounds__(32, 8)
void head_scores(const __half* __restrict__ Q, const unsigned char* __restrict__ kv,
                 const int* __restrict__ bt, const int* __restrict__ qo_indptr,
                 const int* __restrict__ kvlens, float* __restrict__ out,
                 float k_scale, float sm_scale, long page, int bt_cols, int H,
                 int HQ, int NP, int DD, int SD, int ntiles) {
  const int r = blockIdx.x, qh = blockIdx.y, t = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int qo0 = qo_indptr[r];
  const int q_len = qo_indptr[r + 1] - qo0;
  const int kv_len = kvlens[r];
  const int kh = qh / (HQ / H);
  const int base = t * BN;
  float best = -1e30f;
  if (base < kv_len && q_len > 0) {
    const int half = lane >> 4;
    const int l = lane & 15;
    const int qs = qo0 + q_len - 1;         // the last query row of the chunk
    float mx = -1e30f, sm = 0.f;
    for (int ki = 0; ki < 8; ki++) {
      const int tok = min(base + ki * 8 + 4, kv_len - 1);
      const long pg = (long)bt[r * bt_cols + (tok >> 4)] * page;
      const int nn = tok & 15;
      const unsigned char* kd = kv + pg + (long)kh * (NP * DD) + nn * DD + l * 8;
      const unsigned char* ks = kv + pg + (long)H * NP * DD + (long)kh * (NP * SD)
                                + nn * SD + l;
      uint2 v = *reinterpret_cast<const uint2*>(kd);
      const float sf = e4m3(ks[0]) * k_scale;
      float kd16[16];
      const unsigned int w2[2] = {v.x, v.y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
        unsigned int by = w2[w];
#pragma unroll
        for (int bb = 0; bb < 4; bb++) {
          unsigned char b = (by >> (bb * 8)) & 0xFF;
          kd16[w * 8 + bb * 2 + 0] = e2m1(b & 0xF) * sf;
          kd16[w * 8 + bb * 2 + 1] = e2m1((b >> 4) & 0xF) * sf;
        }
      }
      const __half* qp = Q + ((long)qs * HQ + qh) * HD + l * 16;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < 16; d++) acc += __half2float(qp[d]) * kd16[d];
#pragma unroll
      for (int o = 8; o; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o, 16);
      if (half == 0) {
        const float sc = acc * sm_scale;
        const float m2 = fmaxf(mx, sc);
        sm = sm * __expf(mx - m2) + __expf(sc - m2);
        mx = m2;
      }
    }
    best = mx + __logf(fmaxf(sm, 1e-30f));
  }
  if (lane == 0) out[((long)r * HQ + qh) * ntiles + t] = best;
}

static bool g_attr = false;
void run_impl(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
         torch::Tensor qo_indptr, torch::Tensor kvlens, torch::Tensor out,
         double k_scale, double v_scale, double sm_scale, int64_t page,
         int64_t side_off, int64_t bt_cols, int64_t window,
         int64_t max_q_len, c10::optional<torch::Tensor> keep) {
  const char* win_env = std::getenv("VLLM_SM70_PREFILL_WINDOW");
  const char* sink_env = std::getenv("VLLM_SM70_PREFILL_SINK");
  const int win_v = (win_env && atoi(win_env) > 0) ? atoi(win_env) : (int)window;
  const int sink_v = (sink_env && atoi(sink_env) > 0) ? atoi(sink_env) : 0;
  const int R = bt.size(0), HQ = q.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9;          // 16  = D/16  (F = D/2 + D/16)
  const int DD = F - SD;         // 128 = D/2
  const int H = kv.size(1) / 2;
  int smem = (BM * QS + BN * KVS) * 2 + BM * SSS * 4 + BM * SS * 2 + 3 * BM * 4
             + 256 * (int)sizeof(float2);
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
      (int)bt_cols, win_v, sink_v,
      (keep.has_value() && keep->numel())
          ? reinterpret_cast<const unsigned int*>(keep->data_ptr<int>())
          : nullptr,
      (keep.has_value() && keep->dim() == 2) ? (int)keep->size(1) : 0,
      H, HQ, NP, DD, SD, F);
}

void run(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
         torch::Tensor qo_indptr, torch::Tensor kvlens, torch::Tensor out,
         double k_scale, double v_scale, double sm_scale, int64_t page,
         int64_t side_off, int64_t bt_cols, int64_t window,
         int64_t max_q_len) {
  run_impl(q, kv, bt, qo_indptr, kvlens, out, k_scale, v_scale, sm_scale, page,
           side_off, bt_cols, window, max_q_len, c10::nullopt);
}

void run_keep(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
              torch::Tensor qo_indptr, torch::Tensor kvlens, torch::Tensor out,
              double k_scale, double v_scale, double sm_scale, int64_t page,
              int64_t side_off, int64_t bt_cols, int64_t window,
              int64_t max_q_len, torch::Tensor keep) {
  run_impl(q, kv, bt, qo_indptr, kvlens, out, k_scale, v_scale, sm_scale, page,
           side_off, bt_cols, window, max_q_len, keep);
}

void head_stats(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
                torch::Tensor qo_indptr, torch::Tensor kvlens,
                torch::Tensor out, double k_scale, double sm_scale,
                int64_t page, int64_t bt_cols, int64_t ntiles) {
  const int R = bt.size(0), HQ = q.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9, DD = F - SD;
  const int H = kv.size(1) / 2;
  dim3 g(R, HQ, (int)ntiles);
  head_scores<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), qo_indptr.data_ptr<int>(),
      kvlens.data_ptr<int>(), out.data_ptr<float>(), (float)k_scale,
      (float)sm_scale, (long)page, (int)bt_cols, H, HQ, NP, DD, SD, (int)ntiles);
}

void select_blocks(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
            torch::Tensor qo_indptr, torch::Tensor kvlens,
            torch::Tensor scores, torch::Tensor radius, torch::Tensor keep,
            double k_scale,
            double sm_scale, int64_t page, int64_t bt_cols, int64_t max_q_len,
            int64_t tau_milli, int64_t sink_tiles, int64_t local_tiles,
            int64_t tau_rescue_milli, int64_t rescue_k_milli) {
  const int R = bt.size(0), HQ = q.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9, DD = F - SD;
  const int H = kv.size(1) / 2;
  const int ntiles = (int)scores.size(1);
  const int words = (int)keep.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 g(R, H, ntiles);
  sel_scores<<<g, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), qo_indptr.data_ptr<int>(),
      kvlens.data_ptr<int>(), scores.data_ptr<float>(),
      radius.data_ptr<float>(), (float)k_scale,
      (float)sm_scale, (long)page, (int)bt_cols, H, HQ, NP, DD, SD, ntiles);
  dim3 g2(R, H);
  sel_keep<<<g2, 128, 256 * sizeof(float), stream>>>(
      scores.data_ptr<float>(), radius.data_ptr<float>(), keep.data_ptr<int>(),
      kvlens.data_ptr<int>(), (float)tau_milli / 1000.f,
      (float)tau_rescue_milli / 1000.f, (float)rescue_k_milli / 1000.f,
      (int)sink_tiles, (int)local_tiles, ntiles, words);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run);
  m.def("run_keep", &run_keep); m.def("select_blocks", &select_blocks);
  m.def("head_stats", &head_stats); }

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
    # Verified geometries: TP1 = (H=2, HQ=16), TP2 = (H=1, HQ=8). The kernel takes
    # H/HQ as runtime params (grid.y = HQ, kh = qh / (HQ / H)), so both work; other
    # splits stay on the Triton path until they are measured.
    if D != 256 or (H, q.shape[1]) not in ((2, 16), (1, 8)):
        return False
    if max_q_len <= 0 or block_table.shape[0] != qo_indptr.shape[0] - 1:
        return False
    if os.environ.get("VLLM_SM70_PREFILL_HEADSTAT") == "1":
        try:
            _head_stats(q, kv_cache, block_table, qo_indptr, kv_lens, sm_scale,
                        max_q_len)
        except Exception as exc:  # noqa: BLE001
            global _HS_ERR
            _HS_ERR = _HS_ERR + 1 if "_HS_ERR" in globals() else 1
            if _HS_ERR <= 2:
                import traceback
                print("[SM70HEADS] failed: %r" % (exc,), flush=True)
                traceback.print_exc()
    full = os.environ.get("VLLM_SM70_PREFILL_FULL_HEADS", "")
    if full:
        keep = _head_keep(kv_lens, q.shape[1], full, window_left, max_q_len, q.device)
        _mod().run_keep(q, kv_cache, block_table, qo_indptr, kv_lens, out,
                        k_scale, v_scale, sm_scale, kv_cache.stride(0),
                        H * kv_cache.shape[2] * kv_cache.shape[3],
                        block_table.shape[1], window_left, max_q_len, keep)
        return True
    alpha = float(os.environ.get("VLLM_SM70_PREFILL_SELECT_ALPHA", "0") or 0)
    if alpha > 0.0 and os.environ.get("VLLM_SM70_PREFILL_SELECT_TAU_FILE"):
        try:
            with open(os.environ["VLLM_SM70_PREFILL_SELECT_TAU_FILE"]) as fh:
                alpha = float(fh.read().strip() or alpha)
        except Exception:  # noqa: BLE001
            pass
    if alpha <= 0.0:
        _mod().run(q, kv_cache, block_table, qo_indptr, kv_lens, out,
                   k_scale, v_scale, sm_scale, kv_cache.stride(0),
                   H * kv_cache.shape[2] * kv_cache.shape[3],
                   block_table.shape[1], window_left, max_q_len)
        return True
    keep = _select_keep(q, kv_cache, block_table, qo_indptr, kv_lens, alpha,
                        k_scale, sm_scale, max_q_len)
    _mod().run_keep(q, kv_cache, block_table, qo_indptr, kv_lens, out,
                    k_scale, v_scale, sm_scale, kv_cache.stride(0),
                    H * kv_cache.shape[2] * kv_cache.shape[3],
                    block_table.shape[1], window_left, max_q_len, keep)
    return True



def _env_or_file(name, default):
    """int env var, overridable at runtime through <NAME>_FILE (debug sweeps)."""
    path = os.environ.get(name + "_FILE")
    if path:
        try:
            with open(path) as fh:
                return int(float(fh.read().strip()))
        except Exception:  # noqa: BLE001
            pass
    return int(os.environ.get(name, str(default)))


def _env_float_file(name, default):
    path = os.environ.get(name + "_FILE")
    if path:
        try:
            with open(path) as fh:
                return float(fh.read().strip())
        except Exception:  # noqa: BLE001
            pass
    return float(os.environ.get(name, str(default)))



_HS = {}


def _head_stats(q, kv_cache, block_table, qo_indptr, kv_lens, sm_scale, max_q_len):
    """Calibration dump: per (q head, tile) LSE block scores, accumulated over
    layers/chunks, written once to /tmp/headstat.pt for offline analysis."""
    R = block_table.shape[0]
    HQ = q.shape[1]
    ntiles = 4096
    st = _HS
    if st.get("buf") is None or st["buf"].device != q.device:
        st["buf"] = torch.zeros(R * HQ * ntiles, dtype=torch.float32, device=q.device)
        st["acc"] = torch.zeros(R * HQ * ntiles, dtype=torch.float32, device=q.device)
        st["n"] = 0
    _mod().head_stats(q, kv_cache, block_table, qo_indptr, kv_lens,
                      st["buf"][:R * HQ * ntiles].view(R * HQ, ntiles),
                      1.0, sm_scale, kv_cache.stride(0), block_table.shape[1],
                      ntiles)
    st["acc"][:R * HQ * ntiles].add_(st["buf"][:R * HQ * ntiles])
    kv = int(kv_lens[0].item())
    if kv < 32768:
        return
    st["n"] += 1
    if st["n"] % 40 != 1:
        return
    nkv = max((kv + 63) // 64, 1)
    torch.save({"scores": st["buf"][:HQ * ntiles].view(HQ, ntiles).float().cpu(),
                "kv": kv, "nkv": nkv, "calls": st["n"]}, "/tmp/headstat.pt")


_HK = {}


def _head_keep(kv_lens, HQ, full_heads, window_left, max_q_len, device):
    """DuoAttention-style per-head bitmap: listed q heads see everything, the
    rest keep only sink + the recent window (cached per kv length)."""
    full = {int(x) for x in full_heads.split(",") if x.strip() != ""}
    sink = _env_or_file("VLLM_SM70_PREFILL_SELECT_SINK", 32)
    local = _env_or_file("VLLM_SM70_PREFILL_SELECT_LOCAL", 64)
    nkv = int((int(kv_lens[0].item()) + 63) // 64)
    words = (nkv + 31) // 32
    ck = (nkv, tuple(sorted(full)), sink, local)
    if _HK.get("key") != ck:
        bm = torch.zeros((HQ, words), dtype=torch.int32)
        for t in range(nkv):
            keep_t = t < sink or t >= nkv - local
            for h in range(HQ):
                if h in full or keep_t:
                    bm[h, t >> 5] |= 1 << (t & 31)
        _HK["bm"] = bm.to(device)
        _HK["key"] = ck
    return _HK["bm"]

_SEL = {}


def _select_keep(q, kv_cache, block_table, qo_indptr, kv_lens, alpha, k_scale,
                 sm_scale, max_q_len):
    """Block selection for sparse prefill (env VLLM_SM70_PREFILL_SELECT_ALPHA).

    Bitmap is per (request, kv head); tiles below ``sink`` and above
    ``nkv - local`` are always kept, the rest by score >= alpha * max.
    """
    R = block_table.shape[0]
    H = kv_cache.shape[1] // 2
    words = 128                      # 128 * 32 * 64 = 262144 tokens
    ntiles = words * 32
    buf = _SEL.get("buf")
    if buf is None or buf[0].numel() < R * H * ntiles or buf[0].device != q.device:
        buf = (torch.zeros(R * H * ntiles, dtype=torch.float32, device=q.device),
               torch.zeros(R * H * ntiles, dtype=torch.float32, device=q.device),
               torch.zeros(R * H * words, dtype=torch.int32, device=q.device))
        _SEL["buf"] = buf
    scores, radius, keep = buf
    _mod().select_blocks(q, kv_cache, block_table, qo_indptr, kv_lens,
                  scores[:R * H * ntiles].view(R * H, ntiles),
                  radius[:R * H * ntiles].view(R * H, ntiles),
                  keep[:R * H * words].view(R * H, words),
                  k_scale, sm_scale, kv_cache.stride(0), block_table.shape[1],
                  max_q_len, int(round(alpha * 1000)),
                  _env_or_file("VLLM_SM70_PREFILL_SELECT_SINK", 32),
                  _env_or_file("VLLM_SM70_PREFILL_SELECT_LOCAL", 256),
                  int(round(_env_float_file(
                      "VLLM_SM70_PREFILL_SELECT_TAU_RESCUE", 0.0) * 1000)),
                  int(round(_env_float_file(
                      "VLLM_SM70_PREFILL_SELECT_RESCUE_K", 1.2) * 1000)))
    if os.environ.get("VLLM_SM70_PREFILL_SELECT_DEBUG") == "1":
        _SEL["n"] = _SEL.get("n", 0) + 1
        if _SEL["n"] % 200 == 1:
            kv = kv_lens[:R].to(torch.int64)
            nkv = ((kv + 63) // 64).clamp(min=1)
            bits = torch.tensor([bin(int(w) & 0xFFFFFFFF).count("1")
                                 for w in keep[:R * H * words].cpu()], dtype=torch.float32)
            per_call = words
            tot = bits.view(R * H, per_call).sum(1)
            need = torch.cat([nkv[i].repeat(H) for i in range(R)]).float().cpu()
            print(f"[SM70SEL] density={float((tot / need).mean()):.3f} "
                  f"nkv_max={int(nkv.max())} call#{_SEL['n']}", flush=True)
    return keep[:R * H * words].view(R * H, words)
