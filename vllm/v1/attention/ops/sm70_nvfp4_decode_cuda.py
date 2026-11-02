# SPDX-License-Identifier: Apache-2.0
"""Hand-written SM70 NVFP4 decode kernel (scalar FMA, split-KV) as a faster
alternative to the Triton sm70_nvfp4_paged_decode. Env-gated (default off);
shape-gated to (D=256,H=2,HQ=16,N=16); falls back to Triton otherwise.
CG-safe: reuses the caller-provided float workspace for split partials."""
import os
import torch
from torch.utils.cpp_extension import load_inline


def _rows_ok(q: torch.Tensor, nqt: int, bt: torch.Tensor) -> bool:
    """The row-offset path needs rows == reqs*nqt with one block-table row per
    request; padded capture batches take the per-position copy path instead."""
    rows = int(q.shape[0])
    return (
        int(nqt) > 1
        and rows % int(nqt) == 0
        and rows // int(nqt) == int(bt.shape[0])
    )


_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float e2m1(int n){
  const float L[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float v = L[n & 7];
  return (n & 8) ? -v : v;
}
__device__ __forceinline__ float e4m3(unsigned char b){
  int s = (b & 0x80) ? -1 : 1;
  int e = (b >> 3) & 0xF, m = b & 7;
  float v = (e==0) ? (m * 0.001953125f) : ((1.f + m*0.125f) * exp2f((float)(e-7)));
  return s * v;
}

// D=256, RATIO=8, N=16, DD=128, SD=16, F=144
// Geometry is a per-build constant (two dedicated instantiations):
//   TP1: H=2 HQ=16 PAGE=9216   TP2: H=1 HQ=8 PAGE=4608
#ifndef H_DEF
#define H_DEF 2
#endif
#ifndef HQ_DEF
#define HQ_DEF 16
#endif
#ifndef PAGE_DEF
#define PAGE_DEF 9216
#endif
#define D 256
#define H H_DEF
#define HQ HQ_DEF
#define PAGE PAGE_DEF
#define RATIO 8
#define N 16
#define DD 128
#define SD 16
#define F 144

// grid (R, H, splits), block 256 (8 warps = 8 q-heads)
// ws layout: [ ((r*H+h)*RATIO + qh) * splits + sp ][D+2]  (fp32: acc[D], m, l)
__global__ void split_decode(
    const __half* __restrict__ q,     // [R, HQ, D]
    const unsigned char* __restrict__ kv, // [B, 2H, N, F]
    const int* __restrict__ bt,       // [R, bt_cols]
    const int* __restrict__ kvlens,   // [R]
    float* __restrict__ ws,
    float k_scale, float v_scale, float sm_scale,
    int bt_cols, int splits, int window)
{
  const int r = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;   // warp = q-head
  const int kv_len = kvlens[r];
  const long wslot = (long)(((r*H + h)*RATIO + warp) * splits + sp);
  const int wsbase = wslot * (D + 2);

  const long qbase = (long)(r*HQ + h*RATIO + warp) * D;
  // lane owns dims [lane*8, lane*8+8)
  const int d0 = lane * 8;
  float qreg[8];
  #pragma unroll
  for (int j=0;j<8;j++) qreg[j] = __half2float(q[qbase + d0 + j]);
  float acc[8] = {0,0,0,0,0,0,0,0};
  float m_i = -1e30f, l_i = 0.f;

  int lo = (window >= 0) ? max(kv_len - 1 - window, 0) : 0;
  int b_lo = lo / N, nb_hi = (kv_len + N - 1) / N;
  int span = nb_hi - b_lo;
  int b0 = b_lo + (span * sp) / splits;
  int b1 = b_lo + (span * (sp + 1)) / splits;

  __shared__ __half Ks[N][D];
  __shared__ __half Vs[N][D];
  __shared__ float red[8][33];

  for (int b = b0; b < b1; b++) {
    int bid = bt[r*bt_cols + b];
    const long po = (long)bid * PAGE;
    // cooperative dequant: thread -> token (tid/16), sub (tid%16) -> 8 data bytes = 16 dims
    int tok = tid >> 4, sub = tid & 15;
    if (tok < N) {
      int dim0 = sub * 16;
      const unsigned char* kd = kv + po + h*(N*DD) + tok*DD + sub*8;
      const unsigned char* ks = kv + po + H*N*DD + h*(N*SD) + tok*SD + sub; // scale group = sub
      const unsigned char* vd = kv + po + H*N*F + h*(N*DD) + tok*DD + sub*8;
      const unsigned char* vs = kv + po + H*N*F + H*N*DD + h*(N*SD) + tok*SD + sub;
      float sfk = e4m3(*ks) * k_scale;
      float sfv = e4m3(*vs) * v_scale;
      // Vectorized 8-byte load (offsets are 8-aligned) instead of 8 byte loads -> less L1.
      uint2 k2 = *reinterpret_cast<const uint2*>(kd);
      uint2 v2 = *reinterpret_cast<const uint2*>(vd);
      unsigned int kbytes[2] = {k2.x, k2.y};
      unsigned int vbytes[2] = {v2.x, v2.y};
      #pragma unroll
      for (int w=0;w<2;w++){
        #pragma unroll
        for (int b=0;b<4;b++){
          int idx = w*4 + b;
          unsigned char kb = (unsigned char)((kbytes[w] >> (b*8)) & 0xFF);
          Ks[tok][dim0 + idx*2]   = __float2half(e2m1(kb & 0xF) * sfk);
          Ks[tok][dim0 + idx*2+1] = __float2half(e2m1((kb>>4)&0xF) * sfk);
          unsigned char vb = (unsigned char)((vbytes[w] >> (b*8)) & 0xFF);
          Vs[tok][dim0 + idx*2]   = __float2half(e2m1(vb & 0xF) * sfv);
          Vs[tok][dim0 + idx*2+1] = __float2half(e2m1((vb>>4)&0xF) * sfv);
        }
      }
    }
    __syncthreads();
    // this warp (q-head) attends to the N tokens
    #pragma unroll
    for (int t=0;t<N;t++){
      int absolute = b*N + t;
      if (absolute < lo || absolute >= kv_len) continue;
      float part = 0.f;
      #pragma unroll
      for (int j=0;j<8;j++) part += qreg[j] * __half2float(Ks[t][d0+j]);
      // warp reduce
      #pragma unroll
      for (int off=16; off>0; off>>=1) part += __shfl_down_sync(0xffffffff, part, off);
      float sc = __shfl_sync(0xffffffff, part, 0) * sm_scale;
      float m_new = fmaxf(m_i, sc);
      float p = __expf(sc - m_new);
      float alpha = __expf(m_i - m_new);
      l_i = l_i * alpha + p;
      #pragma unroll
      for (int j=0;j<8;j++) acc[j] = acc[j]*alpha + p * __half2float(Vs[t][d0+j]);
      m_i = m_new;
    }
    __syncthreads();
  }
  #pragma unroll
  for (int j=0;j<8;j++) ws[wsbase + d0 + j] = acc[j];
  if (lane == 0) { ws[wsbase + D] = m_i; ws[wsbase + D + 1] = l_i; }
}

// combine: grid (R, H, RATIO), block 256; each block reduces `splits` partials for one (r,h,qh)
__global__ void combine_decode(const float* __restrict__ ws, __half* __restrict__ out,
                               int splits) {
  int r = blockIdx.x, h = blockIdx.y, qh = blockIdx.z;
  int tid = threadIdx.x;
  float m = -1e30f;
  for (int sp=0; sp<splits; sp++){
    float mm = ws[(long)(((r*H+h)*RATIO+qh)*splits + sp)*(D+2) + D];
    m = fmaxf(m, mm);
  }
  float denom = 0.f;
  for (int sp=0; sp<splits; sp++) denom += __expf(ws[(long)(((r*H+h)*RATIO+qh)*splits+sp)*(D+2)+D] - m) * ws[(long)(((r*H+h)*RATIO+qh)*splits+sp)*(D+2)+D+1];
  for (int d=tid; d<D; d+=blockDim.x){
    float acc=0.f;
    for (int sp=0; sp<splits; sp++){
      long base=(long)(((r*H+h)*RATIO+qh)*splits+sp)*(D+2);
      float w=__expf(ws[base+D]-m);
      acc += w * ws[base+d];
    }
    out[(long)(r*HQ + h*RATIO + qh)*D + d] = __float2half(acc / (denom>0?denom:1.f));
  }
}

void run(torch::Tensor q, torch::Tensor kv, torch::Tensor bt, torch::Tensor kvlens,
         torch::Tensor ws, torch::Tensor out, double k_scale, double v_scale,
         double sm_scale, int splits, int window, int R) {

  dim3 g(R, H, splits);
  split_decode<<<g, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      ws.data_ptr<float>(), (float)k_scale, (float)v_scale, (float)sm_scale,
      bt.size(1), splits, window);
  dim3 gc(R, H, RATIO);
  combine_decode<<<gc, 256, 0, at::cuda::getCurrentCUDAStream()>>>(ws.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), splits);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.def("run", &run); }
"""

_MOD = None
_BUF = None       # persistent scratch (tmpO/tM/tL), allocated at init (CG-safe)
_MAXR = 0
_NPARTS = 240     # fixed num_parts for the CUDA path (CG requires fixed)


# --- SM70 XQA-shaped nvfp4 decode (env: VLLM_SM70_DECODE_XQA=1) -------------
# 16-token page per stage, 8 q heads packed into the mma M rows, 256-entry
# byte->half2 smem LUT for nvfp4 decode, ~31 KB smem and <=85 regs so three
# CTAs share an SM. Offline gate (rel<=5e-4) and per-layer ms live in
# sm70/10_reference_study.md 15.8 / sm70_tests/bench_dec_xqa16_lut.py.
# Per-rank part count.  MEASURED 2026-10-30: 120 is exact (worst rel 6.1e-4 over
# kv=1k..25k x 3 repeats on one V100); 240 (the 3-CTA x 80-SM x 2-rank figure)
# is a RACE on a single rank -- rel up to 8e+1 and NaN, non-deterministic, and it
# is what produced the "ductduct..." degenerate output in service.  Override with
# VLLM_SM70_DECODE_XQA_PARTS only when re-measuring the topology.
_NPARTS_XQA = int(os.environ.get("VLLM_SM70_DECODE_XQA_PARTS", "120"))
_NPARTS_BOUNDED_MAX = 4096  # bounded (sink+window) mode: one page per partition
_XQA_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
using namespace nvcuda;

#ifndef H_DEF
#define H_DEF 2
#endif
#ifndef HQ_DEF
#define HQ_DEF 16
#endif
#define HD 256
#define H H_DEF
#define HQ HQ_DEF
#define G 8
#define BN 16
#define QS 264
#define KVS 264
#define SS 16
#define THREADS 256
#define NP 16
#define KDD 128
#define KSD 16

__device__ __forceinline__ float e4m3(unsigned char b) {
  int s = (b & 0x80) ? -1 : 1, e = (b >> 3) & 0xF, m = b & 7;
  if (e == 15 && m == 7) return 448.f;
  float v = (e == 0) ? (m * 0.001953125f) : ((1.f + m * 0.125f) * exp2f((float)(e - 7)));
  return s * v;
}

/* e2m1 -> fp16 bits, no table: e=0 is subnormal (0 / 0.5), else exponent 14+e. */
__device__ __forceinline__ __half2 fp4_pair(unsigned int byte) {
  const unsigned int lo = byte & 0xF, hi = (byte >> 4) & 0xF;
  const unsigned int lm = lo & 7, hm = hi & 7;
  unsigned int a = ((lm >> 1) == 0) ? ((lm & 1) ? 0x3800u : 0u)
                                    : ((((lm >> 1) + 14) << 10) | ((lm & 1) << 9));
  unsigned int b = ((hm >> 1) == 0) ? ((hm & 1) ? 0x3800u : 0u)
                                    : ((((hm >> 1) + 14) << 10) | ((hm & 1) << 9));
  a |= (lo & 8) << 12;
  b |= (hi & 8) << 12;
  return __halves2half2(__ushort_as_half((unsigned short)a),
                        __ushort_as_half((unsigned short)b));
}

#ifndef KVFP8
#define KVFP8 0
#endif
#ifndef KVFP16
#define KVFP16 0
#endif

#if KVFP8
/* 16 fp8 bytes (16 dims) -> 16 halves at dst[0..16) times sc, via a 256-entry
   code -> fp16 table (the fp8 cache stores one byte per element, no packed
   pairs, so each byte is one value). */
__device__ __forceinline__ void deq16(__half* dst, uint4 raw, float sc,
                                      const __half* __restrict__ lut) {
  const unsigned char* b8 = reinterpret_cast<const unsigned char*>(&raw);
#pragma unroll
  for (int k = 0; k < 16; k++) dst[k] = __hmul(lut[b8[k]], __float2half(sc));
}
#endif

#if KVFP16
/* 16 fp16 values (32 bytes) -> 16 scaled halves: no LUT, no unpacking. */
__device__ __forceinline__ void deq16h(__half* dst,
                                       const unsigned char* __restrict__ p,
                                       float sc) {
  const __half2 s2 = __float2half2_rn(sc);
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 16);
  __half2* d = reinterpret_cast<__half2*>(dst);
  const __half2* ha = reinterpret_cast<const __half2*>(&a);
  const __half2* hb = reinterpret_cast<const __half2*>(&b);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    d[k] = __hmul2(ha[k], s2);
    d[4 + k] = __hmul2(hb[k], s2);
  }
}
#endif

/* 8 packed bytes (16 dims) -> 16 halves at dst[d0 .. d0+16) times sc.
   One shared-memory lookup per value pair beats ten ALU ops per pair. */
__device__ __forceinline__ void deq8(__half* dst, uint2 raw, __half2 sc,
                                     const __half2* lut) {
  unsigned int w[2] = {raw.x, raw.y};
#pragma unroll
  for (int j = 0; j < 2; j++) {
    uint4 o;
    unsigned int* ow = reinterpret_cast<unsigned int*>(&o);
#pragma unroll
    for (int b = 0; b < 4; b++) {
      __half2 h = __hmul2(lut[(w[j] >> (b * 8)) & 0xFFu], sc);
      ow[b] = *reinterpret_cast<unsigned int*>(&h);
    }
    *reinterpret_cast<uint4*>(dst + j * 8) = o;
  }
}

__global__ __launch_bounds__(THREADS, 3)
void decode_xqa16(const __half* __restrict__ q, const unsigned char* __restrict__ kv,
                  const int* __restrict__ bt, const int* __restrict__ kvlens,
                  float* __restrict__ ws, float k_scale, float v_scale,
                  float sm_scale, int bt_cols, int nparts, int window,
                  long page, long side_off, const int* __restrict__ page_list,
                  int npages_list, const __half* __restrict__ lut, int hstride,
                  int tstride, int npg, int q0, int nqt) {
  extern __shared__ char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sK = sQ + 16 * QS;
  __half* sV = sK + BN * KVS;
  float* sSp = reinterpret_cast<float*>(sV + BN * KVS);        /* 4 x 16 x 16 */
  __half* sP = reinterpret_cast<__half*>(sSp + 4 * 16 * 16);   /* 16 x SS */
  float* sM = reinterpret_cast<float*>(sP + 16 * SS);
  float* sL = sM + 16;
  float* sA = sL + 16;
  __half2* sLUT = reinterpret_cast<__half2*>(sA + 16);   /* 256 x half2 = 1 KB */

  const int r = blockIdx.z, sp = blockIdx.x, kvh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kv_len = kvlens[r];

  for (int i = tid; i < 16 * (HD / 8); i += THREADS) {
    int row = i / (HD / 8), c8 = (i % (HD / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < G) {
      long off = ((long)(r * nqt + q0) * HQ + kvh * G + row) * HD + c8;
      v = *reinterpret_cast<const uint4*>(q + off);
    }
    *reinterpret_cast<uint4*>(&sQ[row * QS + c8]) = v;
  }
  if (tid < 16) { sM[tid] = -1e30f; sL[tid] = 0.f; sA[tid] = 1.f; }
#if !KVFP8
  sLUT[tid] = fp4_pair(tid);
#endif
  for (int i = tid; i < 16 * SS; i += THREADS) sP[i] = __float2half(0.f);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[2];
  wmma::fill_fragment(o[0], 0.f);
  wmma::fill_fragment(o[1], 0.f);
  /* wmma m16n16 accumulator row of element pair i: this is the mapping that
     validated at rel=4e-4 in the 32-token kernel (not lane>>2). */
  const int rbase = (lane & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);

  const int lo = (window >= 0) ? max(kv_len - 1 - window, 0) : 0;
  int t0, t1, nst;
  int ip0 = 0, ip1 = 0;
  if (page_list != nullptr) {
    /* bounded read: the partition is a slice of the selected page list, one
       16-token page per stage (the last page is clamped by kv_len). */
    ip0 = (int)((long)npages_list * sp / nparts);
    ip1 = (int)((long)npages_list * (sp + 1) / nparts);
    nst = ip1 - ip0;
    t0 = 0;
    t1 = kv_len;
  } else {
    const int span = max(kv_len - lo, 0);
    t0 = lo + (int)((long)span * sp / nparts);
    t1 = lo + (int)((long)span * (sp + 1) / nparts);
    nst = (t1 > t0) ? ((t1 - t0 + BN - 1) / BN) : 0;
  }

  const long kdata0 = (long)kvh * (npg * KDD);
  const long ksf0 = (long)H * npg * KDD + (long)kvh * (npg * KSD);
  const int stok = tid >> 4, ssub = tid & 15;   /* 16 tokens x 16 dims */

  uint2 pk, pv;
  unsigned char kb0 = 0, kb1 = 0, vb0 = 0, vb1 = 0;
#if KVFP8
  uint4 pk4 = make_uint4(0, 0, 0, 0), pv4 = make_uint4(0, 0, 0, 0);
#endif
#if KVFP16
  const unsigned char* kp16 = nullptr;
  const unsigned char* vp16 = nullptr;
#endif
  auto fetch = [&](int st) {
    const int gtok = t0 + st * BN + stok;
    if (page_list != nullptr ? (ip0 + st < ip1) : (gtok < t1)) {
      const int pg = (page_list != nullptr) ? page_list[ip0 + st]
                                            : (gtok / npg);
      const int nn = (page_list != nullptr) ? stok : (gtok % npg);
#if KVFP16
      const long po = (long)bt[r * bt_cols + pg] * page;
      kp16 = kv + po + (long)kvh * hstride + (long)nn * tstride + ssub * 32;
      vp16 = kp16 + HD * 2;
#elif KVFP8
      const long po = (long)bt[r * bt_cols + pg] * page;
      const unsigned char* kp =
          kv + po + (long)kvh * hstride + (long)nn * tstride + ssub * 16;
      pk4 = *reinterpret_cast<const uint4*>(kp);
      pv4 = *reinterpret_cast<const uint4*>(kp + HD);
#else
      const long po = (long)bt[r * bt_cols + pg] * page;
      const unsigned char* kd = kv + po + kdata0 + nn * KDD + ssub * 8;
      const unsigned char* ks = kv + po + ksf0 + nn * KSD + ssub;
      pk = *reinterpret_cast<const uint2*>(kd);
      kb0 = ks[0];
      pv = *reinterpret_cast<const uint2*>(kd + side_off);
      vb0 = ks[side_off];
#endif
    }
  };
  auto commit = [&](int st) {
    const int gtok = t0 + st * BN + stok;
    const int pgc = (page_list != nullptr) ? page_list[ip0 + st]
                                           : (gtok / npg);
    const int nnc = (page_list != nullptr) ? stok : (gtok % npg);
    const bool live = (page_list != nullptr)
                          ? (pgc * BN + nnc < kv_len)
                          : (gtok < t1);
    __half* kd = sK + stok * KVS;
    __half* vd = sV + stok * KVS;
    if (live) {
#if KVFP16
      deq16h(kd + ssub * 16, kp16, k_scale);
      deq16h(vd + ssub * 16, vp16, v_scale);
#elif KVFP8
      deq16(kd + ssub * 16, pk4, k_scale, lut);
      deq16(vd + ssub * 16, pv4, v_scale, lut);
#else
      deq8(kd + ssub * 16, pk, __float2half2_rn(e4m3(kb0) * k_scale), sLUT);
      deq8(vd + ssub * 16, pv, __float2half2_rn(e4m3(vb0) * v_scale), sLUT);
#endif
    } else {
#if KVFP16
      const __half zero = __float2half(0.f);
      __half2* kdz = reinterpret_cast<__half2*>(kd + ssub * 16);
      __half2* vdz = reinterpret_cast<__half2*>(vd + ssub * 16);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        kdz[k] = __half2half2(zero);
        vdz[k] = __half2half2(zero);
      }
#elif KVFP8
      deq16(kd + ssub * 16, make_uint4(0, 0, 0, 0), 0.f, lut);
      deq16(vd + ssub * 16, make_uint4(0, 0, 0, 0), 0.f, lut);
#else
      deq8(kd + ssub * 16, make_uint2(0, 0), __float2half2_rn(0.f), sLUT);
      deq8(vd + ssub * 16, make_uint2(0, 0), __float2half2_rn(0.f), sLUT);
#endif
    }
  };

  fetch(0);
  commit(0);
  __syncthreads();

  for (int st = 0; st < nst; ++st) {
    if (st + 1 < nst) fetch(st + 1);   /* LDGs overlap QK/softmax/PV */

    /* QK: 4 warps, each 4 of the 16 k-steps -> partial score tiles */
    if (warp < 4) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> sa;
      wmma::fill_fragment(sa, 0.f);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int kk = warp * 4 + k;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
        wmma::load_matrix_sync(qf, &sQ[kk * 16], QS);
        wmma::load_matrix_sync(kf, &sK[kk * 16], KVS);
        wmma::mma_sync(sa, qf, kf, sa);
      }
      wmma::store_matrix_sync(&sSp[warp * 256], sa, 16, wmma::mem_row_major);
    }
    __syncthreads();

    /* softmax: one warp per two heads (lanes 0-15 = head 2w, 16-31 = head 2w+1)
       so no lane sits idle and the warp count halves. */
    if (warp < 4) {
      const int head = 2 * warp + (lane >> 4);
      const int tl = lane & 15;
      const int at = t0 + st * BN + tl;
      const int pgt = (page_list != nullptr) ? page_list[ip0 + st]
                                             : (at / npg);
      const int nnt = (page_list != nullptr) ? tl : (at % npg);
      const int gtok_t = pgt * BN + nnt;
      float sc = -1e30f;
      if ((page_list != nullptr) ? (gtok_t < kv_len) : (at < t1)) {
        sc = (sSp[head * 16 + tl] + sSp[256 + head * 16 + tl] +
              sSp[512 + head * 16 + tl] + sSp[768 + head * 16 + tl]) * sm_scale;
      }
      float mx = sc;
#pragma unroll
      for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      const float m_old = sM[head];
      const float m_new = fmaxf(m_old, mx);
      const float p = __expf(sc - m_new);
      float lsum = p;
#pragma unroll
      for (int off = 8; off; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
      sP[head * SS + tl] = __float2half(p);
      if (tl == 0) {   /* running stats are per head, so only lane 0 needs alpha */
        const float alpha = (m_old <= -1e29f) ? 0.f : __expf(m_old - m_new);
        sM[head] = m_new;
        sL[head] = sL[head] * alpha + lsum;
        sA[head] = alpha;
      }
    }
    __syncthreads();

    /* PV: 8 warps x 2 accumulators (16 dims each), one 16-token k-step */
    {
      const float a0 = sA[rbase], a2 = sA[rbase + 2];
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
      wmma::load_matrix_sync(pf, sP, SS);   /* the same P serves both halves */
#pragma unroll
      for (int t = 0; t < 2; t++) {
        float* acc = o[t].x;
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] *= ((j & 2) ? a2 : a0);
        const int nd = warp * 2 + t;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
        wmma::load_matrix_sync(vf, &sV[nd * 16], KVS);
        wmma::mma_sync(o[t], pf, vf, o[t]);
      }
    }
    __syncthreads();
    if (st + 1 < nst) { commit(st + 1); }
    __syncthreads();
  }

  {
  float* scr = reinterpret_cast<float*>(sK);
#pragma unroll
    for (int t = 0; t < 2; t++)
      wmma::store_matrix_sync(scr + (warp * 2 + t) * 256, o[t], 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < G * HD; i += THREADS) {
      const int row = i / HD, d = i % HD;
      const int wsrc = d / 32, tsrc = (d % 32) / 16, c = d % 16;
      const long rb = (long)(((r * H + kvh) * G + row) * nparts + sp) * (HD + 2);
      ws[rb + d] = scr[(wsrc * 2 + tsrc) * 256 + row * 16 + c];
    }
    if (tid < G) {
      const long rb = (long)(((r * H + kvh) * G + tid) * nparts + sp) * (HD + 2);
      ws[rb + HD] = sM[tid];
      ws[rb + HD + 1] = sL[tid];
    }
  }
}

__global__ void combine_decode(const float* __restrict__ ws,
                               __half* __restrict__ out, int nparts, int q0,
                               int nqt) {
  int r = blockIdx.x, h = blockIdx.y, qh = blockIdx.z;
  int tid = threadIdx.x;
  float m = -1e30f;
  for (int sp = 0; sp < nparts; sp++)
    m = fmaxf(m, ws[(long)(((r * H + h) * G + qh) * nparts + sp) * (HD + 2) + HD]);
  float denom = 0.f;
  for (int sp = 0; sp < nparts; sp++)
    denom += __expf(ws[(long)(((r * H + h) * G + qh) * nparts + sp) * (HD + 2) + HD] - m) *
             ws[(long)(((r * H + h) * G + qh) * nparts + sp) * (HD + 2) + HD + 1];
  for (int d = tid; d < HD; d += blockDim.x) {
    float acc = 0.f;
    for (int sp = 0; sp < nparts; sp++) {
      long b = (long)(((r * H + h) * G + qh) * nparts + sp) * (HD + 2);
      acc += __expf(ws[b + HD] - m) * ws[b + d];
    }
    out[(long)((r * nqt + q0) * HQ + h * G + qh) * HD + d] =
        __float2half(acc / (denom > 0 ? denom : 1.f));
  }
}

static bool g_attr = false;

void prepare() {
  const int smem = (16 * QS + 2 * BN * KVS) * 2 + 4 * 16 * 16 * 4 + 16 * SS * 2 + 16 * 12 + 1024;
  cudaFuncSetAttribute(decode_xqa16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  g_attr = true;
}

void run(torch::Tensor q, torch::Tensor kv, torch::Tensor bt, torch::Tensor kvlens,
         torch::Tensor ws, torch::Tensor out, double k_scale, double v_scale,
         double sm_scale, int nparts, int window, int R,
         c10::optional<torch::Tensor> page_list,
         c10::optional<torch::Tensor> lut = c10::nullopt,
         int64_t hstride = 0, int64_t tstride = 0, int64_t npg = 0,
         int64_t q0 = 0, int64_t nqt = 1) {
#if KVFP8 || KVFP16
  // fp8 / fp16 caches are [blocks, H, N, 2D]: dim 1 is the real kv heads.
  const int H2 = kv.size(1);
#else
  const int H2 = kv.size(1) / 2;
#endif
  const long page = kv.stride(0);
  const int smem = (16 * QS + 2 * BN * KVS) * 2 + 4 * 16 * 16 * 4 + 16 * SS * 2 + 16 * 12 + 1024;
  if (!g_attr) {
    cudaFuncSetAttribute(decode_xqa16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g_attr = true;
  }
  dim3 g(nparts, H2, R);
  decode_xqa16<<<g, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      ws.data_ptr<float>(), (float)k_scale, (float)v_scale, (float)sm_scale,
      bt.size(1), nparts, window, page, page / 2,
      page_list.has_value() ? page_list->data_ptr<int>() : nullptr,
      page_list.has_value() ? (int)page_list->numel() : 0,
      lut.has_value() ? reinterpret_cast<const __half*>(lut->data_ptr<at::Half>())
                      : nullptr,
      (int)hstride, (int)tstride, (int)npg, (int)q0, (int)nqt);
  dim3 gc(R, H2, G);
  combine_decode<<<gc, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      ws.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      nparts, (int)q0, (int)nqt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); m.def("prepare", &prepare); }
"""

_MODS_XQA = {}
_PAGE_I = None
_PAGE_BUF = None
_SEL_BT = None
_SEL_LENS = None


def _mod_xqa(H=2, HQ=16):
    key = (int(H), int(HQ))
    if key not in _MODS_XQA:
        _MODS_XQA[key] = load_inline(
            name=f"sm70_nvfp4_decode_xqa_h{key[0]}q{key[1]}",
            cpp_sources="", cuda_sources=_XQA_CUDA, verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               f"-DH_DEF={key[0]}", f"-DHQ_DEF={key[1]}"])
    return _MODS_XQA[key]


_MODS_XQA_FP8 = {}


def _mod_xqa_fp8(H=2, HQ=16):
    """fp8 KV variant of the XQA decode (same kernel, KVFP8 build)."""
    key = (int(H), int(HQ))
    if key not in _MODS_XQA_FP8:
        _MODS_XQA_FP8[key] = load_inline(
            name=f"sm70_fp8_decode_xqa_h{key[0]}q{key[1]}",
            cpp_sources="", cuda_sources=_XQA_CUDA, verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-DKVFP8=1",
                               f"-DH_DEF={key[0]}", f"-DHQ_DEF={key[1]}"])
    return _MODS_XQA_FP8[key]


_MODS = {}


def _mod(H=2, HQ=16):
    """Dedicated instantiation per GQA geometry (compile-time H/HQ/PAGE).

    TP1 = (2, 16, page 9216), TP2 = (1, 8, page 4608). Sharing one build with
    runtime H/HQ/PAGE cost 9% on TP1 decode (12.79 -> 13.97 ms), so each geometry
    keeps its own constants.
    """
    global _MOD
    key = (int(H), int(HQ))
    if key not in _MODS:
        page = 2 * int(H) * 16 * 144
        _MODS[key] = load_inline(
            name=f"sm70_nvfp4_decode_cuda_v2_h{key[0]}q{key[1]}",
            cpp_sources="", cuda_sources=_CUDA, verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               f"-DH_DEF={key[0]}", f"-DHQ_DEF={key[1]}",
                               f"-DPAGE_DEF={page}"])
    _MOD = _MODS[key]
    return _MOD


def warmup_compile(device=None, max_reqs=8):
    """Compile the module AND allocate the persistent 240-part scratch buffer,
    both BEFORE CUDA-graph capture (compiling/allocating inside capture breaks
    the graph)."""
    global _BUF, _MAXR
    if os.environ.get("VLLM_SM70_DECODE_XQA") == "1":
        _mod_xqa(2, 16).prepare()
        _mod_xqa(1, 8).prepare()
    _mod(2, 16)
    _mod(1, 8)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _MAXR = max(int(max_reqs), 1)
    _bnd = int(os.environ.get("VLLM_SM70_DECODE_SINK_PAGES", "0")) + int(
        os.environ.get("VLLM_SM70_DECODE_WINDOW_PAGES", "0")
    )
    _slots = max(_NPARTS, min(_bnd, _NPARTS_BOUNDED_MAX)) if _bnd else _NPARTS
    n = _MAXR * 2 * 8 * _slots * 258  # R*H*RATIO*NPARTS*(D+2)
    if _BUF is None or _BUF.numel() < n or _BUF.device != device:
        _BUF = torch.zeros(n, dtype=torch.float32, device=device)


_MODS_XQA_FP16 = {}


def _mod_xqa_fp16(H=2, HQ=16):
    """fp16 KV variant of the XQA decode (same kernel, KVFP16 build)."""
    key = (int(H), int(HQ))
    if key not in _MODS_XQA_FP16:
        _MODS_XQA_FP16[key] = load_inline(
            name=f"sm70_fp16_decode_xqa_h{key[0]}q{key[1]}",
            cpp_sources="", cuda_sources=_XQA_CUDA, verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-DKVFP16=1",
                               f"-DH_DEF={key[0]}", f"-DHQ_DEF={key[1]}"])
    return _MODS_XQA_FP16[key]


def paged_decode_fp8(q, kv_cache, block_table, kv_lens, out, k_scale, v_scale,
                     sm_scale, window, splits, workspace, kv_dtype="fp8_e4m3",
                     nqt=1):
    """SM70 fp8 KV decode: in-kernel LUT dequant, no shadow buffer, no FA2.

    Same split-KV XQA structure as the NVFP4 path; the KV reader is the
    ``KVFP8`` build of the kernel (one byte per element, 256-entry code table).
    Graph-safe: all inputs are plain tensors, no per-step host work.
    ``nqt`` > 1 is spec-decode verify (see ``paged_decode_fp16``).
    """
    if nqt > 1:
        R = q.shape[0] // nqt
        base = kv_lens - (nqt - 1)
        for p in range(nqt):
            out_p = torch.empty(R, q.shape[1], q.shape[2], dtype=out.dtype,
                                device=out.device)
            paged_decode_fp8(
                q[p::nqt].contiguous(), kv_cache, block_table, base + p, out_p,
                k_scale, v_scale, sm_scale, window, splits, workspace,
                kv_dtype=kv_dtype,
            )
            out[p::nqt] = out_p
        return
    from vllm.v1.attention.ops.sm70_nvfp4_dequant import _fp8_lut



    R, HQ, D = q.shape
    H = int(kv_cache.shape[1])
    kv_u8 = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
    lut = _fp8_lut(kv_u8.device, kv_dtype)
    mod = _mod_xqa_fp8(H, HQ)
    mod.prepare()
    mod.run(q.contiguous(), kv_u8, block_table, kv_lens, workspace, out,
            float(k_scale), float(v_scale), float(sm_scale), int(splits),
            int(window), R, None, lut, kv_u8.stride(1), kv_u8.stride(2),
            int(kv_u8.size(2)))


def paged_decode_fp16(q, kv_cache, block_table, kv_lens, out, k_scale, v_scale,
                      sm_scale, window_left, splits, workspace, nqt=1,
                      q0=0, _single=False):
    """SM70 fp16 KV decode: same split-KV XQA, no dequant at all.

    ``nqt`` > 1 is spec-decode verify: each request owns ``nqt`` consecutive
    query rows and position p attends causally up to its own token. The kernel
    is launched once per position with a shortened ``kv_lens`` (the draft
    tokens are already in the cache), leaving the q=1 fast path untouched.
    """
    if nqt > 1 and not _single:
        base = kv_lens - (nqt - 1)
        if _rows_ok(q, nqt, block_table):
            for p in range(nqt):
                paged_decode_fp16(
                    q, kv_cache, block_table, base + p, out,
                    k_scale, v_scale, sm_scale, window_left, splits, workspace,
                    q0=p, nqt=nqt, _single=True,
                )
            return
        R = q.shape[0] // nqt
        for p in range(nqt):
            out_p = torch.empty(R, q.shape[1], q.shape[2], dtype=out.dtype,
                                device=out.device)
            paged_decode_fp16(
                q[p::nqt].contiguous(), kv_cache, block_table, base + p, out_p,
                k_scale, v_scale, sm_scale, window_left, splits, workspace,
            )
            out[p::nqt] = out_p
        return
    R, HQ, D = q.shape
    R = R // int(nqt)
    H = int(kv_cache.shape[1])
    kv_u8 = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
    mod = _mod_xqa_fp16(H, HQ)
    mod.prepare()
    mod.run(q.contiguous(), kv_u8, block_table, kv_lens, workspace, out,
            float(k_scale), float(v_scale), float(sm_scale), int(splits),
            int(window_left), R, None, None, kv_u8.stride(1),
            kv_u8.stride(2), int(kv_u8.size(2)), int(q0), int(nqt))


def try_paged_decode_cuda(q, kv_cache, block_table, kv_lens, out,
                          k_scale, v_scale, sm_scale, window_left, splits, workspace,
                          nqt=1, q0=0):
    if os.environ.get("VLLM_SM70_DECODE_CUDA") != "1":
        return False
    if q.dtype != torch.float16 or q.dim() != 3:
        return False
    R, HQ, D = q.shape
    R = R // int(nqt)
    H = kv_cache.shape[1] // 2
    # Verified geometries: TP1 (H=2, HQ=16), TP2 (H=1, HQ=8); ratio stays 8.
    if D != 256 or (H, HQ) not in ((2, 16), (1, 8)):
        return False
    if int(kv_cache.stride(0)) != 2 * H * 16 * 144:
        return False
    if _BUF is None or R > _MAXR:
        return False
    q = q.contiguous()
    if not out.is_contiguous():
        return False
    if os.environ.get("VLLM_SM70_DECODE_DEBUG") == "1":
        global _DEC_DBG
        try:
            _DEC_DBG += 1
        except NameError:
            _DEC_DBG = 1
        if _DEC_DBG <= 3:
            print(f"[SM70DECDBG] q={tuple(q.shape)} kv={tuple(kv_cache.shape)} "
                  f"kv_stride0={kv_cache.stride(0)} bt={tuple(block_table.shape)} "
                  f"kvlens={kv_lens.tolist()} splits={splits} "
                  f"window={window_left}", flush=True)
    if os.environ.get("VLLM_SM70_DECODE_XQA") == "1":
        need = R * H * 8 * _NPARTS_XQA * 258
        _sink = int(os.environ.get("VLLM_SM70_DECODE_SINK_PAGES", "0"))
        _win = int(os.environ.get("VLLM_SM70_DECODE_WINDOW_PAGES", "0"))
        if _sink or _win:
            # Flash-Next's approach, training-free: attend to a *bounded* set of
            # pages instead of the whole cache.  Rather than teaching the kernel
            # a page list (which costs one CTA prologue per page), the selected
            # pages are COMPACTED into a short block table so the normal dense
            # path runs over ~10% of the tokens with full efficiency.
            # Everything below is device-side and uses static buffers, so a FULL
            # cudagraph can hold it.
            global _SEL_BT, _SEL_LENS
            npg = torch.div(kv_lens[:1] + 15, 16, rounding_mode="floor")      # [1]
            k_sink = torch.clamp(npg, max=_sink)
            w_start = torch.clamp(npg - _win, min=k_sink.to(torch.int32))
            n_sel = k_sink + (npg - w_start)
            cap = _sink + _win
            if _SEL_BT is None or _SEL_BT.numel() < cap:
                _SEL_BT = torch.zeros(
                    (1, cap), dtype=block_table.dtype, device=block_table.device
                )
                _SEL_LENS = torch.zeros(
                    (1,), dtype=kv_lens.dtype, device=kv_lens.device
                )
            idx = torch.arange(cap, device=block_table.device, dtype=torch.int32)
            page = torch.where(
                idx < k_sink,
                idx,
                torch.minimum(w_start + (idx - k_sink), npg - 1),
            )
            valid = idx < n_sel
            src = block_table[0].gather(0, page.to(torch.int64).clamp_(max=block_table.shape[1] - 1))
            _SEL_BT[0].copy_(torch.where(valid, src, torch.zeros_like(src)))
            # keep the last page's true token count so the tail stays masked
            tail = kv_lens[:1] - (npg - 1) * 16
            _SEL_LENS[:1].copy_(torch.where(n_sel > 0, (n_sel - 1) * 16 + tail, kv_lens[:1]))
            need = R * H * 8 * _NPARTS_XQA * 258
            _mod_xqa(H, HQ).run(q, kv_cache, _SEL_BT, _SEL_LENS, _BUF[:need], out,
                                k_scale, v_scale, sm_scale, _NPARTS_XQA,
                                -1, R, None, None, 0, 0,
                                int(kv_cache.size(2)))
            return True
        _mod_xqa(H, HQ).run(q, kv_cache, block_table, kv_lens, _BUF[:need], out,
                            k_scale, v_scale, sm_scale, _NPARTS_XQA,
                            int(window_left), R, None, None, 0, 0,
                            int(kv_cache.size(2)), int(q0), int(nqt))
        return True
    need = R * H * 8 * _NPARTS * 258
    _mod(H, HQ).run(q, kv_cache, block_table, kv_lens, _BUF[:need], out,
               k_scale, v_scale, sm_scale, _NPARTS, int(window_left), R, None,
               None, 0, 0, int(kv_cache.size(2)), int(q0), int(nqt))
    return True
