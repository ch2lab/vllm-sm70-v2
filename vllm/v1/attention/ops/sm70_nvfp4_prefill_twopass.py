# SPDX-License-Identifier: Apache-2.0
"""SM70 paged-nvfp4 prefill through the two-pass (GEMM + fused epilogue) path.

fa_pf (hand-written wmma) runs 24-25 TFLOPS because every mma step reloads its
fragments from shared memory; cuBLAS reaches 65.3 TFLOPS on the QK shape.  The
blocked two-pass with a fused softmax epilogue measures

    fa_pf        169-176 ms / 24-25 TFLOPS   (kv=65536, q=4096)
    two-pass     120.2 ms  / 35.4 TFLOPS     (same input, rel 3.47e-4)

Layout of this module (identical to the verified prototype):
  * Python owns every buffer and loops over (request, kv head, q block);
  * C++ gets one such tile and runs the whole k loop:
      bridge_kv (paged nvfp4 -> fp16, one 64-token block at a time, reused buffer)
      -> at::mm_out  (QK, cuBLAS, contiguous S)
      -> fused_softmax (mask + rowmax + exp + rowsum, P in place)
      -> at::mm_out  (PV)
      -> fused_update (O = O*alpha + PV)
  * the mask is position-based only, so it is prefix-cache safe.

Valve: VLLM_SM70_PREFILL_TWOPASS (default 0 = off -> fa_pf stays in charge).

Block sparsity: VLLM_SM70_PREFILL_BLKSPARSE=<percent of eligible k blocks to keep>
(default 0 = off).  run_tiles_sparse bridges every k block once, scores it with the
Quest upper bound (max over the block of q_mean . k * sm_scale), then runs QK /
softmax / PV only for the kept (q tile, k block) pairs.  Sink and the tile's own
diagonal/recent blocks are always kept.  At keep=100% the output is bit-identical
to run_tiles; at keep=40% a 250K prefill drops from 146.6 s to 96.7 s while the
canary (7/7) and 131K/250K retrieval (3/3) gates still pass.
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
using namespace at::cuda;

#define DH 256

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

// Paged nvfp4 K/V for one k block of one request and one kv head -> dense fp16.
// Thread tid owns one 16-byte group (32 dims) of one token; 8 groups per token.
#ifndef KVFP8
#define KVFP8 0
#endif
#ifndef KVFP16
#define KVFP16 0
#endif

/* fp8 -> fp16 through a 256-entry code table (one byte per element). */
__device__ __forceinline__ __half2 f8_pair(unsigned char b0, unsigned char b1,
                                           float sc, const __half* lut) {
  return __floats2half2_rn(__half2float(lut[b0]) * sc,
                           __half2float(lut[b1]) * sc);
}

__global__ void bridge_kv(const unsigned char* __restrict__ kv,
                          const int* __restrict__ bt, __half* __restrict__ Kb,
                          __half* __restrict__ Vb, int r, int kh, int k0, int bn,
                          int kv_len, long page, long side_off, int bt_cols,
                          int H, int NP, int DD, int SD, float k_scale,
                          float v_scale, const __half* __restrict__ lut,
                          int hstride, int tstride) {
  const int tok = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
  const int sub = threadIdx.x % 8;
  if (tok >= bn) return;
  const int t = k0 + tok;
  if (t >= kv_len) return;                  // tail: leave the buffer as is (masked)
  const int b = t / NP, nn = t % NP;
  const unsigned char* pd = kv + (long)bt[r * bt_cols + b] * page;
#if KVFP16
  /* fp16 cache: [blocks, H, N, 2D], K in [0:D], V in [D:2D], 2 bytes each. */
  const long f16base = (long)kh * hstride + (long)nn * tstride + sub * 64;
  const unsigned char* kp16 = pd + f16base;
  const unsigned char* vp16 = kp16 + 512;
  const __half2 s16k = __float2half2_rn(k_scale);
  const __half2 s16v = __float2half2_rn(v_scale);
  const int d0h = sub * 32;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    union { __half2 h2[4]; uint4 u; } pk, pv;
    const uint4 ku = *reinterpret_cast<const uint4*>(kp16 + w * 16);
    const uint4 vu = *reinterpret_cast<const uint4*>(vp16 + w * 16);
    const __half2* kh2 = reinterpret_cast<const __half2*>(&ku);
    const __half2* vh2 = reinterpret_cast<const __half2*>(&vu);
#pragma unroll
    for (int bb = 0; bb < 4; bb++) {
      pk.h2[bb] = __hmul2(kh2[bb], s16k);
      pv.h2[bb] = __hmul2(vh2[bb], s16v);
    }
    *reinterpret_cast<uint4*>(&Kb[tok * DH + d0h + w * 8]) = pk.u;
    *reinterpret_cast<uint4*>(&Vb[tok * DH + d0h + w * 8]) = pv.u;
  }
  return;
#endif
#if KVFP8
  /* fp8 cache: [blocks, H, N, 2D], K in [0:D], V in [D:2D], one byte each. */
  const long f8base = (long)kh * hstride + (long)nn * tstride + sub * 32;
  const unsigned char* kp = pd + f8base;
  const unsigned char* vp = kp + 256;
  const int d0f = sub * 32;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    union { __half2 h2[4]; uint4 u; } pk, pv;
#pragma unroll
    for (int bb = 0; bb < 4; bb++) {
      const int off = w * 8 + bb * 2;
      pk.h2[bb] = f8_pair(kp[off], kp[off + 1], k_scale, lut);
      pv.h2[bb] = f8_pair(vp[off], vp[off + 1], v_scale, lut);
    }
    *reinterpret_cast<uint4*>(&Kb[tok * DH + d0f + w * 8]) = pk.u;
    *reinterpret_cast<uint4*>(&Vb[tok * DH + d0f + w * 8]) = pv.u;
  }
  return;
#endif
  const long kdata = (long)kh * (NP * DD) + (long)nn * DD + sub * 16;
  const long ksf = (long)H * NP * DD + (long)kh * (NP * SD) + (long)nn * SD + sub * 2;
  const unsigned char* bs = pd + ksf;
  const float ks0 = e4m3(bs[0]) * k_scale, ks1 = e4m3(bs[1]) * k_scale;
  const float vs0 = e4m3(bs[side_off]) * v_scale;
  const float vs1 = e4m3(bs[side_off + 1]) * v_scale;
  const uint4 kd = *reinterpret_cast<const uint4*>(pd + kdata);
  const uint4 vd = *reinterpret_cast<const uint4*>(pd + kdata + side_off);
  const unsigned int kw[4] = {kd.x, kd.y, kd.z, kd.w};
  const unsigned int vw[4] = {vd.x, vd.y, vd.z, vd.w};
  const int d0 = sub * 32;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    const float sfk = (w < 2) ? ks0 : ks1;
    const float sfv = (w < 2) ? vs0 : vs1;
    union { __half2 h2[4]; uint4 u; } pk, pv;
#pragma unroll
    for (int bb = 0; bb < 4; bb++) {
      const unsigned char bk = (kw[w] >> (bb * 8)) & 0xFF;
      const unsigned char bv = (vw[w] >> (bb * 8)) & 0xFF;
      pk.h2[bb] = __floats2half2_rn(e2m1(bk & 0xF) * sfk,
                                    e2m1((bk >> 4) & 0xF) * sfk);
      pv.h2[bb] = __floats2half2_rn(e2m1(bv & 0xF) * sfv,
                                    e2m1((bv >> 4) & 0xF) * sfv);
    }
    *reinterpret_cast<uint4*>(&Kb[tok * DH + d0 + w * 8]) = pk.u;
    *reinterpret_cast<uint4*>(&Vb[tok * DH + d0 + w * 8]) = pv.u;
  }
}

// Mask + row max + exp + rowsum with P written in place.  One warp per row and
// the whole row lives in registers, so each element is read once and written
// once (the 3-pass version read the logits twice and ran ~2x slower).
// PER = BN / 32 halves per lane.
template <int PER>
__global__ void fused_softmax_reg(__half* __restrict__ S,
                                  float* __restrict__ mrow,
                                  float* __restrict__ lrow,
                                  float* __restrict__ alpha,
                                  const int* __restrict__ qpos, int nrows,
                                  int k0, int kv_len, float sm) {
  constexpr int NH = PER / 2;           // __half2 values per lane
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= nrows) return;
  const int pos = qpos[row];
  const int lim = min(pos + 1, kv_len) - k0;   // live columns in this block
  __half* base = S + (long)row * (PER * 32);
  // A warp covers 64 halves per half2 instruction; 4 of those fill one 256-half
  // chunk.  Every lane's 128 B region is thus contiguous with its neighbours.
  auto col = [&](int i) { return (i >> 2) * 256 + (i & 3) * 64 + lane * 2; };
  __half2 r[NH];
#pragma unroll
  for (int i = 0; i < NH; i++)
    r[i] = *reinterpret_cast<const __half2*>(base + col(i));
  float mx = -1e30f;
#pragma unroll
  for (int i = 0; i < NH; i++) {
    const int c = col(i);
    if (c + 1 < lim) mx = fmaxf(mx, fmaxf(__low2float(r[i]), __high2float(r[i])) * sm);
    else if (c < lim) mx = fmaxf(mx, __low2float(r[i]) * sm);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  const float newm = fmaxf(mrow[row], mx);
  const float a = __expf(mrow[row] - newm);
  float ls = 0.f;
#pragma unroll
  for (int i = 0; i < NH; i++) {
    const int c = col(i);
    float e0 = 0.f, e1 = 0.f;
    if (c + 1 < lim) {
      e0 = __expf(__low2float(r[i]) * sm - newm);
      e1 = __expf(__high2float(r[i]) * sm - newm);
    } else if (c < lim) {
      e0 = __expf(__low2float(r[i]) * sm - newm);
    }
    ls += e0 + e1;
    r[i] = __floats2half2_rn(e0, e1);
  }
#pragma unroll
  for (int i = 0; i < NH; i++)
    *reinterpret_cast<__half2*>(base + col(i)) = r[i];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, off);
  if (lane == 0) {
    mrow[row] = newm;
    lrow[row] = lrow[row] * a + ls;
    alpha[row] = a;
  }
}

// Runtime-width variant: used for the causal-diagonal k block, whose useful
// width (columns at or before the tile's last position) is data dependent.
__global__ void fused_softmax_gen(__half* __restrict__ S, float* __restrict__ mrow,
                                  float* __restrict__ lrow,
                                  float* __restrict__ alpha,
                                  const int* __restrict__ qpos, int nrows,
                                  int k0, int bn, int kv_len, float sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= nrows) return;
  const int pos = qpos[row];
  const int lim = min(pos + 1, kv_len) - k0;
  __half* base = S + (long)row * bn;
  const int stride = 64;                     // 32 lanes x 2 halves per step
  float mx = -1e30f;
  for (int c0 = 0; c0 < bn; c0 += stride) {
    const int c = c0 + lane * 2;
    if (c >= bn) continue;
    if (c + 1 < bn && c + 1 < lim) {
      const __half2 h = *reinterpret_cast<const __half2*>(base + c);
      mx = fmaxf(mx, fmaxf(__low2float(h), __high2float(h)) * sm);
    } else if (c < lim) {
      mx = fmaxf(mx, __half2float(base[c]) * sm);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  const float newm = fmaxf(mrow[row], mx);
  const float a = __expf(mrow[row] - newm);
  float ls = 0.f;
  for (int c0 = 0; c0 < bn; c0 += stride) {
    const int c = c0 + lane * 2;
    if (c >= bn) continue;
    float e0 = 0.f, e1 = 0.f;
    const bool pair = (c + 1 < bn);
    if (pair && c + 1 < lim) {
      const __half2 h = *reinterpret_cast<const __half2*>(base + c);
      e0 = __expf(__low2float(h) * sm - newm);
      e1 = __expf(__high2float(h) * sm - newm);
    } else if (c < lim) {
      e0 = __expf(__half2float(base[c]) * sm - newm);
    }
    ls += e0 + e1;
    if (pair) {
      *reinterpret_cast<__half2*>(base + c) = __floats2half2_rn(e0, e1);
    } else {
      base[c] = __float2half(e0);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, off);
  if (lane == 0) {
    mrow[row] = newm;
    lrow[row] = lrow[row] * a + ls;
    alpha[row] = a;
  }
}

// Dispatch on BN; specialised widths keep the row in registers.
#define SM_LAUNCH(BN)                                                        \
  fused_softmax_reg<(BN) / 32><<<(nrows + 3) / 4, 128, 0, stream>>>(          \
      reinterpret_cast<__half*>(S.data_ptr<at::Half>()), mrow.data_ptr<float>(), \
      lrow.data_ptr<float>(), alpha.data_ptr<float>(), qpos.data_ptr<int>(),  \
      nrows, k0, (int)kv_len, (float)sm)

inline void launch_softmax(torch::Tensor S, torch::Tensor mrow,
                           torch::Tensor lrow, torch::Tensor alpha,
                           torch::Tensor qpos, int nrows, int k0, int kv_len,
                           float sm, cudaStream_t stream) {
  switch (S.size(1)) {
    case 2048: SM_LAUNCH(2048); break;
    case 4096: SM_LAUNCH(4096); break;
    case 8192: SM_LAUNCH(8192); break;
    default: TORCH_CHECK(false, "unsupported BN ", S.size(1));
  }
}

__global__ void fused_update(float* __restrict__ O, const __half* __restrict__ PV,
                             const float* __restrict__ alpha, int M) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const float a = alpha[row];
  for (int d = tid; d < DH; d += blockDim.x)
    O[row * DH + d] = O[row * DH + d] * a + __half2float(PV[row * DH + d]);
}

// One (request, kv head, q block).  Q is the q-block slice [nrows, HQ, D] and
// out the matching slice; qpos[i] is the absolute position of row i / G.
void run(torch::Tensor Q, torch::Tensor kv, torch::Tensor bt,
         torch::Tensor qpos, torch::Tensor Kb, torch::Tensor Vb,
         torch::Tensor S, torch::Tensor PV, torch::Tensor O, torch::Tensor mrow,
         torch::Tensor lrow, torch::Tensor alpha, torch::Tensor out,
         double k_scale, double v_scale, double sm_scale, int64_t r,
         int64_t kh, int64_t kv_len, int64_t page, int64_t side_off,
         int64_t bt_cols, int64_t BN) {
  const int HQ = Q.size(1), D = Q.size(2);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9, DD = F - SD;
  const int H = kv.size(1) / 2;
  const int G = HQ / H;
  const int nrows = Q.size(0);
  const int M = nrows * G;
  auto Qb = Q.narrow(1, kh * G, G).contiguous().reshape({M, D});
  // The caller may hand over maximum-sized buffers; trim to this tile.  Rows are
  // the outer dim, so every view below stays contiguous.
  S = S.narrow(0, 0, M);
  PV = PV.narrow(0, 0, M);
  O = O.narrow(0, 0, M);
  mrow = mrow.narrow(0, 0, M);
  lrow = lrow.narrow(0, 0, M);
  alpha = alpha.narrow(0, 0, M);
  qpos = qpos.narrow(0, 0, M);
  auto stream = getCurrentCUDAStream();
  O.zero_();
  mrow.fill_(-1e30);
  lrow.zero_();
  alpha.fill_(1.0);
  for (int k0 = 0; k0 < (int)kv_len; k0 += (int)BN) {
    const int bn = (int)BN;
    const int threads = 256, per = threads / 8;
    bridge_kv<<<(bn + per - 1) / per, threads, 0, stream.stream()>>>(
        kv.data_ptr<unsigned char>(), bt.data_ptr<int>(),
        reinterpret_cast<__half*>(Kb.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Vb.data_ptr<at::Half>()), (int)r, (int)kh, k0,
        bn, (int)kv_len, (long)page, (long)side_off, (int)bt_cols, (int)H,
        (int)NP, (int)DD, (int)SD, (float)k_scale, (float)v_scale, nullptr, 0,
        0);
    at::mm_out(S, Qb, Kb.transpose(0, 1));
    launch_softmax(S, mrow, lrow, alpha, qpos, M, k0, (int)kv_len,
                   (float)sm_scale, stream.stream());
    at::mm_out(PV, S, Vb);
    fused_update<<<M, 128, 0, stream.stream()>>>(
        O.data_ptr<float>(),
        reinterpret_cast<const __half*>(PV.data_ptr<at::Half>()),
        alpha.data_ptr<float>(), M);
  }
  auto res = O.view({nrows, G, D}) / lrow.view({nrows, G, 1});
  out.narrow(1, kh * G, G).copy_(res.to(at::kHalf));
}

// ---- bench-only entry points (used to attribute time between stages) ----------
void bench_softmax(torch::Tensor S, torch::Tensor mrow, torch::Tensor lrow,
                   torch::Tensor alpha, torch::Tensor qpos, int64_t k0,
                   int64_t bn, int64_t kv_len, double sm) {
  launch_softmax(S, mrow, lrow, alpha, qpos, (int)S.size(0), (int)k0,
                 (int)kv_len, (float)sm, getCurrentCUDAStream().stream());
}

void bench_update(torch::Tensor O, torch::Tensor PV, torch::Tensor alpha) {
  const int M = O.size(0);
  fused_update<<<M, 128, 0, getCurrentCUDAStream().stream()>>>(
      O.data_ptr<float>(),
      reinterpret_cast<const __half*>(PV.data_ptr<at::Half>()),
      alpha.data_ptr<float>(), M);
}

void bench_bridge(torch::Tensor kv, torch::Tensor bt, torch::Tensor Kb,
                  torch::Tensor Vb, int64_t r, int64_t kh, int64_t k0,
                  int64_t bn, int64_t kv_len, int64_t page, int64_t side_off,
                  int64_t bt_cols, int64_t H, int64_t NP, int64_t DD,
                  int64_t SD, double k_scale, double v_scale) {
  const int threads = 256, per = threads / 8;
  bridge_kv<<<((int)bn + per - 1) / per, threads, 0,
              getCurrentCUDAStream().stream()>>>(
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(),
      reinterpret_cast<__half*>(Kb.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(Vb.data_ptr<at::Half>()), (int)r, (int)kh,
      (int)k0, (int)bn, (int)kv_len, (long)page, (long)side_off, (int)bt_cols,
      (int)H, (int)NP, (int)DD, (int)SD, (float)k_scale, (float)v_scale, nullptr,
      0, 0);
}

// Same maths as run(), but the k loop is outermost and *all* q tiles of one
// request are handled inside it: the paged->fp16 bridge then runs once per
// (kv head, k block) instead of once per (q tile, kv head), and the bridged K/V
// stays hot in L2 across the tiles.
void run_tiles(torch::Tensor Qd, torch::Tensor kv, torch::Tensor bt,
               torch::Tensor qpos, torch::Tensor Kb, torch::Tensor Vb,
               torch::Tensor S, torch::Tensor PV, torch::Tensor O,
               torch::Tensor mrow, torch::Tensor lrow, torch::Tensor alpha,
               torch::Tensor out, std::vector<int64_t> rows,
               std::vector<int64_t> plast, double k_scale,
               double v_scale, double sm_scale, int64_t r, int64_t kh,
               int64_t kv_len, int64_t page, int64_t side_off, int64_t bt_cols,
               int64_t BN, torch::Tensor lut, int64_t hstride,
               int64_t tstride) {
  const int D = Qd.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9, DD = F - SD;
#if KVFP8 || KVFP16
  const int H = kv.size(1);
#else
  const int H = kv.size(1) / 2;
#endif
  const int HQ = out.size(1);
  const int Gq = HQ / H;
  auto stream = getCurrentCUDAStream();
  for (int k0 = 0; k0 < (int)kv_len; k0 += (int)BN) {
    const int bn = (int)BN;
    const int threads = 256, per = threads / 8;
    bridge_kv<<<(bn + per - 1) / per, threads, 0, stream.stream()>>>(
        kv.data_ptr<unsigned char>(), bt.data_ptr<int>(),
        reinterpret_cast<__half*>(Kb.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Vb.data_ptr<at::Half>()), (int)r, (int)kh, k0,
        bn, (int)kv_len, (long)page, (long)side_off, (int)bt_cols, (int)H,
        (int)NP, (int)DD, (int)SD, (float)k_scale, (float)v_scale,
        reinterpret_cast<const __half*>(lut.data_ptr<at::Half>()),
        (int)hstride, (int)tstride);
    long o = 0;
    for (size_t t = 0; t < rows.size(); t++) {
      const int M = (int)rows[t] * Gq;
      auto St = S.narrow(0, 0, M);
      auto PVt = PV.narrow(0, 0, M);
      // Causal diagonal: rows end at plast[t], so columns beyond it are fully
      // masked -- shrink the block to the useful width instead of computing
      // and then masking the tail.
      // Round the useful width up to 8 halves: fused_softmax_gen reads the row
      // through __half2, so an odd row stride would misalign (and the extra
      // columns are masked anyway).
      int hi = (int)std::min<int64_t>(BN, plast[t] + 1 - k0);
      if (hi <= 0) { o += M; continue; }
      hi = (hi + 7) & ~7;
      if (hi < (int)BN) {
        auto Sd = at::mm(Qd.narrow(0, o, M), Kb.narrow(0, 0, hi).transpose(0, 1));
        fused_softmax_gen<<<(M + 3) / 4, 128, 0, stream.stream()>>>(
            reinterpret_cast<__half*>(Sd.data_ptr<at::Half>()),
            mrow.narrow(0, o, M).data_ptr<float>(),
            lrow.narrow(0, o, M).data_ptr<float>(),
            alpha.narrow(0, o, M).data_ptr<float>(),
            qpos.narrow(0, o, M).data_ptr<int>(), M, k0, hi, (int)kv_len,
            (float)sm_scale);
        auto PVd = at::mm(Sd, Vb.narrow(0, 0, hi));
        fused_update<<<M, 128, 0, stream.stream()>>>(
            O.narrow(0, o, M).data_ptr<float>(),
            reinterpret_cast<const __half*>(PVd.data_ptr<at::Half>()),
            alpha.narrow(0, o, M).data_ptr<float>(), M);
      } else {
        at::mm_out(St, Qd.narrow(0, o, M), Kb.transpose(0, 1));
        launch_softmax(St, mrow.narrow(0, o, M), lrow.narrow(0, o, M),
                       alpha.narrow(0, o, M), qpos.narrow(0, o, M), M, k0,
                       (int)kv_len, (float)sm_scale, stream.stream());
        at::mm_out(PVt, St, Vb);
        fused_update<<<M, 128, 0, stream.stream()>>>(
            O.narrow(0, o, M).data_ptr<float>(),
            reinterpret_cast<const __half*>(PVt.data_ptr<at::Half>()),
            alpha.narrow(0, o, M).data_ptr<float>(), M);
      }
      o += M;
    }
  }
  long o = 0, qoff = 0;
  for (size_t t = 0; t < rows.size(); t++) {
    const int nr = (int)rows[t], M = nr * Gq;
    auto res = O.narrow(0, o, M).view({nr, Gq, D}) /
               lrow.narrow(0, o, M).view({nr, Gq, 1});
    out.narrow(0, qoff, nr).narrow(1, kh * Gq, Gq).copy_(res.to(at::kHalf));
    o += M;
    qoff += nr;
  }
}
// ---- block-sparse prefill (Quest-style upper bound) ---------------------------
// sc[t * nb + b] = max over the block's tokens of dot(Kb[tok], qmean[t]) * sm.
// This is the exact maximum logit any token of block b can reach for tile t, so
// ranking blocks by it never discards a block that could dominate the softmax.
// Grid is (tiles, splits): one block per split keeps every SM busy, since the
// per-call work is otherwise tiny and latency-bound.
__device__ __forceinline__ void atomic_max_f(float* p, float v) {
  int* pi = reinterpret_cast<int*>(p);
  int old = *pi, assumed;
  while (v > __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(pi, assumed, __float_as_int(v));
    if (old == assumed) break;
  }
}

__global__ void blk_max_score(const __half* __restrict__ Kb,
                              const __half* __restrict__ qmean,
                              float* __restrict__ sc, float* __restrict__ scsu,
                              int bslot, int nb, int bn, int D, float sm,
                              int nsplit, int parts) {
  const int t = blockIdx.x, part = blockIdx.y;
  const int per = (bn + nsplit - 1) / nsplit;
  const int lo = part * per;
  const int hi = std::min(bn, lo + per);
  const __half* q = qmean + (long)t * D;
  float mx = -1e30f;
  for (int tok = lo + threadIdx.x; tok < hi; tok += blockDim.x) {
    const __half* k = Kb + (long)tok * D;
    float acc = 0.f;
    for (int d = 0; d < D; d += 2) {
      const float2 qq = __half22float2(*reinterpret_cast<const __half2*>(q + d));
      const float2 kk = __half22float2(*reinterpret_cast<const __half2*>(k + d));
      acc += qq.x * kk.x + qq.y * kk.y;
    }
    mx = fmaxf(mx, acc);
  }
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = -1e30f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) v = fmaxf(v, red[w]);
    v *= sm;
    // sc: block-level (the prefill mask reads this on the host).
    // scsu: split-level = one 256-token superblock, which is the granularity the
    // decode selector ranks.  Split `part` of block `bslot` is superblock
    // bslot * parts + part, so the store is race free.
    atomic_max_f(sc + (long)t * nb + bslot, v);
    if (part < parts) scsu[(long)t * nb * parts + (long)bslot * parts + part] = v;
  }
}

// Copy the deepest q tile's superblock scores into the page-keyed decode prior.
// Pages of one superblock share a score, so the store is a plain one (the last
// write wins: deepest layer, last kv head, last tile = closest to decode).
__global__ void seed_prior(float* __restrict__ prior,
                           const int* __restrict__ bt_row,
                           const float* __restrict__ scsu_row, int nsuper,
                           int pages_per_sb, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nsuper) return;
  const float v = scsu_row[i];
  const int p0 = i * pages_per_sb;
  for (int j = 0; j < pages_per_sb; ++j)
    prior[bt_row[min(p0 + j, cols - 1)]] = v;
}

// Same maths as run_tiles, but only the kept (q tile, k block) pairs run their
// QK/softmax/PV.  Block scores come from one cheap pass that bridges every block
// once and keeps the per-tile maximum logit, so the mask is the Quest bound
// rather than a sampled estimate.  Sink blocks and the tile's own diagonal and
// most recent blocks are always kept.
void run_tiles_sparse(torch::Tensor Qd, torch::Tensor kv, torch::Tensor bt,
                      torch::Tensor qpos, torch::Tensor Kb, torch::Tensor Vb,
                      torch::Tensor S, torch::Tensor PV, torch::Tensor O,
                      torch::Tensor mrow, torch::Tensor lrow,
                      torch::Tensor alpha, torch::Tensor out,
                      std::vector<int64_t> rows, std::vector<int64_t> plast,
                      double k_scale, double v_scale, double sm_scale,
                      int64_t r, int64_t kh, int64_t kv_len, int64_t page,
                      int64_t side_off, int64_t bt_cols, int64_t BN,
                      torch::Tensor lut, int64_t hstride, int64_t tstride,
                      int64_t nsink, int64_t nrecent, double keep_frac,
                      torch::Tensor prior) {
  const int D = Qd.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int SD = F / 9, DD = F - SD;
#if KVFP8 || KVFP16
  const int H = kv.size(1);
#else
  const int H = kv.size(1) / 2;
#endif
  const int HQ = out.size(1);
  const int Gq = HQ / H;
  const int nblk = (int)((kv_len + BN - 1) / BN);
  const int nt = (int)rows.size();
  auto stream = getCurrentCUDAStream();
  auto opts = Qd.options();
  auto qmean = torch::empty({nt, D}, opts);
  {
    auto qf = Qd.to(at::kFloat);
    long o = 0;
    for (int t = 0; t < nt; t++) {
      const int M = (int)rows[t] * Gq;
      qmean[t].copy_(qf.narrow(0, o, M).mean(0));
      o += M;
    }
  }
  auto sc = torch::full({nt, nblk}, -1e30f, Qd.options().dtype(at::kFloat));
  const int threads = 256, per = threads / 8;
  const int nsplit = (int)std::max(1, std::min(32, (int)(BN / 256)));
  auto scsu = torch::full({nt, nblk * nsplit}, -1e30f,
                          Qd.options().dtype(at::kFloat));
  dim3 sgrid((unsigned)nt, (unsigned)nsplit);
  for (int b = 0; b < nblk; b++) {
    const int k0 = b * (int)BN;
    const int bnb = (int)std::min<int64_t>(BN, kv_len - k0);
    bridge_kv<<<(int(BN) + per - 1) / per, threads, 0, stream.stream()>>>(
        kv.data_ptr<unsigned char>(), bt.data_ptr<int>(),
        reinterpret_cast<__half*>(Kb.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Vb.data_ptr<at::Half>()), (int)r, (int)kh, k0,
        (int)BN, (int)kv_len, (long)page, (long)side_off, (int)bt_cols, (int)H,
        (int)NP, (int)DD, (int)SD, (float)k_scale, (float)v_scale,
        reinterpret_cast<const __half*>(lut.data_ptr<at::Half>()),
        (int)hstride, (int)tstride);
    blk_max_score<<<sgrid, 256, 0, stream.stream()>>>(
        reinterpret_cast<const __half*>(Kb.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(qmean.data_ptr<at::Half>()),
        sc.data_ptr<float>(), scsu.data_ptr<float>(), b, nblk, bnb, D,
        (float)sm_scale, nsplit, nsplit);
  }
  if (prior.numel() > 0) {
    // Decode hot set: the deepest q tile sees exactly what the decode query
    // will see, so its per-superblock Quest scores are the ranking to reuse.
    const int nsuper = nblk * nsplit;
    const int pages_per_sb = (int)std::max<int64_t>(1, 256 / kv.size(2));
    seed_prior<<<(nsuper + 255) / 256, 256, 0, stream.stream()>>>(
        prior.data_ptr<float>(), bt.data_ptr<int>(),
        scsu.data_ptr<float>() + (long)(nt - 1) * nsuper, nsuper, pages_per_sb,
        (int)bt_cols);
  }
  auto sch = sc.to(at::kCPU);
  const float* sp = sch.data_ptr<float>();
  std::vector<char> keep((size_t)nt * nblk, 0);
  for (int t = 0; t < nt; t++) {
    const int bmax = (int)(plast[t] / BN);        // last eligible block
    if (bmax < 0) continue;
    const int n_el = std::min(bmax + 1, nblk);
    const int forced = (int)std::min<int64_t>(nsink + nrecent, n_el);
    int budget = forced + (int)std::ceil(keep_frac * (double)(n_el - forced));
    budget = std::min(std::max(budget, forced), n_el);
    std::vector<int> idx;
    idx.reserve(n_el);
    for (int b = 0; b < n_el; b++) {
      const bool f = (b < nsink) || (b >= n_el - nrecent);
      if (f) keep[(size_t)t * nblk + b] = 1;
      else idx.push_back(b);
    }
    const int extra = budget - forced;
    if (extra > 0 && !idx.empty()) {
      std::partial_sort(idx.begin(), idx.begin() + std::min<int>(extra, idx.size()),
                        idx.end(), [&](int a, int b) {
                          return sp[(size_t)t * nblk + a] > sp[(size_t)t * nblk + b];
                        });
      for (int i = 0; i < extra && i < (int)idx.size(); i++)
        keep[(size_t)t * nblk + idx[i]] = 1;
    }
  }
  for (int b = 0; b < nblk; b++) {
    const int k0 = b * (int)BN;
    bool any = false;
    for (int t = 0; t < nt; t++) any = any || keep[(size_t)t * nblk + b];
    if (!any) continue;
    bridge_kv<<<(int(BN) + per - 1) / per, threads, 0, stream.stream()>>>(
        kv.data_ptr<unsigned char>(), bt.data_ptr<int>(),
        reinterpret_cast<__half*>(Kb.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Vb.data_ptr<at::Half>()), (int)r, (int)kh, k0,
        (int)BN, (int)kv_len, (long)page, (long)side_off, (int)bt_cols, (int)H,
        (int)NP, (int)DD, (int)SD, (float)k_scale, (float)v_scale,
        reinterpret_cast<const __half*>(lut.data_ptr<at::Half>()),
        (int)hstride, (int)tstride);
    long o = 0;
    for (int t = 0; t < nt; t++) {
      const int M = (int)rows[t] * Gq;
      int hi = (int)std::min<int64_t>(BN, plast[t] + 1 - k0);
      if (hi <= 0 || !keep[(size_t)t * nblk + b]) { o += M; continue; }
      hi = (hi + 7) & ~7;
      auto St = S.narrow(0, 0, M);
      auto PVt = PV.narrow(0, 0, M);
      if (hi < (int)BN) {
        auto Sd = at::mm(Qd.narrow(0, o, M), Kb.narrow(0, 0, hi).transpose(0, 1));
        fused_softmax_gen<<<(M + 3) / 4, 128, 0, stream.stream()>>>(
            reinterpret_cast<__half*>(Sd.data_ptr<at::Half>()),
            mrow.narrow(0, o, M).data_ptr<float>(),
            lrow.narrow(0, o, M).data_ptr<float>(),
            alpha.narrow(0, o, M).data_ptr<float>(),
            qpos.narrow(0, o, M).data_ptr<int>(), M, k0, hi, (int)kv_len,
            (float)sm_scale);
        auto PVd = at::mm(Sd, Vb.narrow(0, 0, hi));
        fused_update<<<M, 128, 0, stream.stream()>>>(
            O.narrow(0, o, M).data_ptr<float>(),
            reinterpret_cast<const __half*>(PVd.data_ptr<at::Half>()),
            alpha.narrow(0, o, M).data_ptr<float>(), M);
      } else {
        at::mm_out(St, Qd.narrow(0, o, M), Kb.transpose(0, 1));
        launch_softmax(St, mrow.narrow(0, o, M), lrow.narrow(0, o, M),
                       alpha.narrow(0, o, M), qpos.narrow(0, o, M), M, k0,
                       (int)kv_len, (float)sm_scale, stream.stream());
        at::mm_out(PVt, St, Vb);
        fused_update<<<M, 128, 0, stream.stream()>>>(
            O.narrow(0, o, M).data_ptr<float>(),
            reinterpret_cast<const __half*>(PVt.data_ptr<at::Half>()),
            alpha.narrow(0, o, M).data_ptr<float>(), M);
      }
      o += M;
    }
  }
  long o = 0, qoff = 0;
  for (int t = 0; t < nt; t++) {
    const int nr = (int)rows[t], M = nr * Gq;
    auto res = O.narrow(0, o, M).view({nr, Gq, D}) /
               lrow.narrow(0, o, M).view({nr, Gq, 1});
    out.narrow(0, qoff, nr).narrow(1, kh * Gq, Gq).copy_(res.to(at::kHalf));
    o += M;
    qoff += nr;
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("run_tiles", &run_tiles);
  m.def("run_tiles_sparse", &run_tiles_sparse);
  m.def("bench_softmax", &bench_softmax);
  m.def("bench_update", &bench_update);
  m.def("bench_bridge", &bench_bridge);
}
"""

_MOD = None


_MODS: dict = {}

# Below this many (q x kv) products the per-call fixed costs dominate for
# NVFP4; fp8/fp16 have no alternative native reader so they ignore it. Spec
# verify (q = k+1) sets it to 0 to force the two-pass kernel.
_MIN_QK = 1 << 20

_FMT_FLAGS = {"nvfp4": [], "fp8": ["-DKVFP8=1"], "fp16": ["-DKVFP16=1"]}


def _mod(fmt: str = "nvfp4"):
    """Two-pass module; ``fmt`` selects the KV reader (nvfp4 / fp8 / fp16)."""
    if fmt not in _MODS:
        assert fmt in _FMT_FLAGS, fmt
        _MODS[fmt] = load_inline(
            name="sm70_nvfp4_prefill_twopass_v1"
            + ("" if fmt == "nvfp4" else "_" + fmt),
            cpp_sources="", cuda_sources=_CUDA, verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"] + _FMT_FLAGS[fmt])
    return _MODS[fmt]


def enabled() -> bool:
    return os.environ.get("VLLM_SM70_PREFILL_TWOPASS") == "1"


_BUF: dict = {}
_EMPTY = torch.empty(0, dtype=torch.float32)


def _num(name, dflt):
    try:
        return int(os.environ.get(name, dflt))
    except ValueError:
        return dflt


BM_DEF = _num("VLLM_SM70_TWOPASS_BM", 1024)
BN_DEF = _num("VLLM_SM70_TWOPASS_BN", 4096)


def _seed_prior_buf(kv_cache):
    """Page-keyed Quest prior for the decode selector (empty = disabled).

    The decode hot set can then be ranked by the real attention structure seen
    during prefill instead of by a fresh sampled estimate, at zero extra reads.
    """
    if _num("VLLM_SM70_PREFILL_SEED_PRIOR", 0) != 1:
        return _EMPTY
    from .sm70_decode_pagescore_cuda import prior_buffer

    return prior_buffer(int(kv_cache.shape[0]), kv_cache.device)


def _bufs(dev, G, BM, BN, D, need_rows):
    key = (str(dev), G, BM, BN, D)
    b = _BUF.get(key)
    if b is None:
        h = dict(dtype=torch.float16, device=dev)
        f = dict(dtype=torch.float32, device=dev)
        M = BM * G
        b = _BUF[key] = dict(
            Kb=torch.empty(BN, D, **h), Vb=torch.empty(BN, D, **h),
            S=torch.empty(M, BN, **h), PV=torch.empty(M, D, **h), cap=0)
    if b["cap"] < need_rows:                      # grow for more q tiles
        b["cap"] = need_rows
        h = dict(dtype=torch.float16, device=dev)
        f = dict(dtype=torch.float32, device=dev)
        b["Qd"] = torch.empty(need_rows, D, **h)
        b["O"] = torch.zeros(need_rows, D, **f)
        b["mrow"] = torch.full((need_rows,), -1e30, **f)
        b["lrow"] = torch.zeros(need_rows, **f)
        b["alpha"] = torch.ones(need_rows, **f)
        b["qpos"] = torch.zeros(need_rows, dtype=torch.int32, device=dev)
    return b


def try_paged_prefill_twopass(q, kv_cache, block_table, qo_indptr, kv_lens, out,
                              k_scale, v_scale, sm_scale, window_left,
                              max_q_len, kv_fp8=False, kv_fp16=False,
                              lut=None) -> bool:
    """Run the two-pass path for this prefill call; False = fall back."""
    if os.environ.get("VLLM_SM70_TWOPASS_DEBUG") == "1":
        logger.info(
            "twopass gate: enabled=%s win=%s q=%s/%s out=%s/%s kv=%s D=%s "
            "H=%s HQ=%s maxq=%s bt=%s indptr=%s",
            enabled(), window_left, tuple(q.shape), q.dtype,
            tuple(out.shape), out.dtype, tuple(kv_cache.shape),
            q.shape[2] if q.dim() == 3 else -1,
            kv_cache.shape[1] if kv_cache.dim() == 4 else -1,
            q.shape[1] if q.dim() == 3 else -1, max_q_len,
            tuple(block_table.shape), tuple(qo_indptr.shape),
        )
    if not enabled() or window_left >= 0:
        return False
    if not (q.is_cuda and q.dtype == torch.float16 and q.is_contiguous()
            and out.is_contiguous()):
        return False
    D = q.shape[2]
    H = (kv_cache.shape[1] if (kv_fp8 or kv_fp16)
         else kv_cache.shape[1] // 2)
    HQ = q.shape[1]
    if D != 256 or (H, HQ) not in ((2, 16), (1, 8)):
        return False
    if max_q_len <= 0 or block_table.shape[0] != qo_indptr.shape[0] - 1:
        return False
    G = HQ // H
    BM, BN = _num("VLLM_SM70_TWOPASS_BM", BM_DEF), _num("VLLM_SM70_TWOPASS_BN",
                                                        BN_DEF)
    side_off = H * kv_cache.shape[2] * kv_cache.shape[3]
    try:
        qo, kl = qo_indptr.tolist(), kv_lens.tolist()
        if os.environ.get("VLLM_SM70_PROBE_MASS"):
            # Task 2.5.1 probe: the last token of the chunk attends causally to
            # the whole prefix, which is exactly the distribution the decode step
            # will face.  Prefill is never CUDA-graph captured, so this always
            # runs.  Any failure disables the probe, never the request.
            try:
                from . import sm70_attn_mass_cuda as _mass

                if _mass.enabled():
                    _mass.probe(qo, q, kv_cache, block_table, kv_lens,
                                sm_scale, 128, 8192)
            except Exception:  # noqa: BLE001
                os.environ.pop("VLLM_SM70_PROBE_MASS", None)
        for r in range(block_table.shape[0]):
            q0, q1 = int(qo[r]), int(qo[r + 1])
            klen = int(kl[r])
            qlen = q1 - q0
            # Small tiles are dominated by the bridge and fixed overheads,
            # but fp8/fp16 KV has no other native reader (stock flashinfer
            # paged kernels are not sm_70-safe at head_dim 256), so every
            # shape must go through here.
            if qlen * klen < _MIN_QK and not (kv_fp8 or kv_fp16):
                return False
            rows = [min(BM, qlen - i) for i in range(0, qlen, BM)]
            b = _bufs(q.device, G, BM, BN, D, sum(rows) * G)
            Qd, O, mrow, lrow, alpha, qpos = (
                b["Qd"], b["O"], b["mrow"], b["lrow"], b["alpha"], b["qpos"])
            for kh in range(H):
                off = 0
                for t, nr in enumerate(rows):
                    M = nr * G
                    Qd[off:off + M].copy_(
                        q[q0 + t * BM:q0 + t * BM + nr, kh * G:(kh + 1) * G]
                        .reshape(M, D))
                    base = klen - qlen + t * BM
                    qpos[off:off + M].copy_(
                        torch.arange(base, base + nr, dtype=torch.int32,
                                     device=q.device).repeat_interleave(G))
                    off += M
                plast = [klen - qlen + t * BM + nr - 1
                         for t, nr in enumerate(rows)]
                tot = off
                O[:tot].zero_()
                mrow[:tot].fill_(-1e30)
                lrow[:tot].zero_()
                alpha[:tot].fill_(1.0)
                _fmt = "fp8" if kv_fp8 else ("fp16" if kv_fp16 else "nvfp4")
                bs = _num("VLLM_SM70_PREFILL_BLKSPARSE", 0)
                if bs > 0:
                    prior = _seed_prior_buf(kv_cache)
                    _mod(_fmt).run_tiles_sparse(
                        Qd, kv_cache, block_table, qpos, b["Kb"], b["Vb"],
                        b["S"], b["PV"], O, mrow, lrow, alpha, out[q0:q1],
                        rows, plast, k_scale, v_scale, sm_scale, r, kh, klen,
                        kv_cache.stride(0), side_off, block_table.shape[1], BN,
                        lut if lut is not None else q, kv_cache.stride(1),
                        kv_cache.stride(2),
                        _num("VLLM_SM70_PREFILL_SINK_BLK", 1),
                        _num("VLLM_SM70_PREFILL_RECENT_BLK", 1),
                        bs / 100.0, prior)
                else:
                    _mod(_fmt).run_tiles(Qd, kv_cache, block_table, qpos, b["Kb"],
                                     b["Vb"], b["S"], b["PV"], O, mrow, lrow,
                                     alpha, out[q0:q1], rows, plast, k_scale,
                                     v_scale,
                                     sm_scale, r, kh, klen, kv_cache.stride(0),
                                     side_off, block_table.shape[1], BN,
                                     lut if lut is not None else q,
                                     kv_cache.stride(1), kv_cache.stride(2))
        return True
    except Exception as e:  # noqa: BLE001 - the valve must never break prefill
        if os.environ.get("VLLM_SM70_TWOPASS_DEBUG") == "1":
            raise
        logger.warning_once("SM70 two-pass prefill fell back: %r", e)
        return False
