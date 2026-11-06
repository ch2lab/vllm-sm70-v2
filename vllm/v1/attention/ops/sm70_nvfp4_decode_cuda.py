# SPDX-License-Identifier: Apache-2.0
"""Hand-written SM70 NVFP4 decode kernel (scalar FMA, split-KV) as a faster
alternative to the Triton sm70_nvfp4_paged_decode. Env-gated (default off);
shape-gated to (D=256,H=2,HQ=16,N=16); falls back to Triton otherwise.
CG-safe: reuses the caller-provided float workspace for split partials."""
import os
import torch
from torch.utils.cpp_extension import load_inline

from .sm70_attention_geometry import supported_decode


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

// D=256, N=16, DD=128, SD=16, F=144
// Geometry is a per-build constant (one instantiation per TP layout):
//   TP1: H=2 HQ=16 PAGE=9216   TP2: H=1 HQ=8 PAGE=4608   TP4: H=1 HQ=4 PAGE=4608
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
#define RATIO (HQ / H)
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
  const bool active = warp < RATIO;
  const long wslot = (long)(((r*H + h)*RATIO + (active ? warp : 0)) * splits + sp);
  const int wsbase = wslot * (D + 2);

  const long qbase = (long)(r*HQ + h*RATIO + (active ? warp : 0)) * D;
  // lane owns dims [lane*8, lane*8+8)
  const int d0 = lane * 8;
  float qreg[8];
  if (active) {
    #pragma unroll
    for (int j=0;j<8;j++) qreg[j] = __half2float(q[qbase + d0 + j]);
  }
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
    // this warp (q-head) attends to the N tokens; idle warps only help dequant
    if (active) {
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
    }
    __syncthreads();
  }
  if (active) {
    #pragma unroll
    for (int j=0;j<8;j++) ws[wsbase + d0 + j] = acc[j];
    if (lane == 0) { ws[wsbase + D] = m_i; ws[wsbase + D + 1] = l_i; }
  }
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
  // 256 threads are required by the cooperative dequant (16 tokens x 16 lanes);
  // only the first RATIO warps own a q head.
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
#define G (HQ / H)
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

/* Partitions below this many tokens cost more in partial traffic than in the
   work they carry: at kv=10K, 240 parts move ~7.9 MB of fp32 partials against
   5.9 MB of actual KV, and the kernel measured 34% slower than 120 parts
   (sm70_tests/bench_xqa_parts.py: 73.3 us at 240 vs 48.6 us at 120).  85
   tokens per part is that measured optimum, and it leaves 250K untouched
   (249936/85 >> 240, so the cap is inactive there and 240 still wins). */
constexpr int kMinTokensPerPart = 85;

__global__ __launch_bounds__(THREADS, 3)
void decode_xqa16_pair(const __half* __restrict__ q, const unsigned char* __restrict__ kv,
                  const int* __restrict__ bt, const int* __restrict__ kvlens,
                  float* __restrict__ ws, float k_scale, float v_scale,
                  float sm_scale, int bt_cols, int nparts, int window,
                  long page, long side_off, const int* __restrict__ page_list,
                  int npages_list, const __half* __restrict__ lut, int hstride,
                  int tstride, int npg, int q0, int nqt) {
  extern __shared__ char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sK = sQ + 2 * 16 * QS;
  __half* sV = sK + BN * KVS;
  float* sSp = reinterpret_cast<float*>(sV + BN * KVS);        /* 4 x 16 x 16 */
  __half* sP = reinterpret_cast<__half*>(sSp + 4 * 16 * 16);   /* 16 x SS */
  float* sM = reinterpret_cast<float*>(sP + 2 * 16 * SS);
  float* sL = sM + 32;
  float* sA = sL + 32;
  __half2* sLUT = reinterpret_cast<__half2*>(sA + 32);   /* 256 x half2 = 1 KB */

  const int r = blockIdx.z, sp = blockIdx.x, kvh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kv_len = kvlens[r] + 1;

  /* How many partitions actually carry work for this request.  Computed here
     and identically in combine_decode, so the two agree on which rows of ws
     are live without any device-side handshake (which would not be safe under
     CUDA-graph capture). */
  const int lo0 = (window >= 0) ? max(kv_len - 1 - window, 0) : 0;
  const int span0 = max(kv_len - lo0, 0);
  const int np = (page_list != nullptr)
                     ? nparts
                     : min(nparts, max(1, span0 / kMinTokensPerPart));
  if (page_list == nullptr && sp >= np) return;

  for (int pp = 0; pp < 2; ++pp) {
    for (int i = tid; i < 16 * (HD / 8); i += THREADS) {
      int row = i / (HD / 8), c8 = (i % (HD / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < G) {
        long off = ((long)(r * nqt + q0 + pp) * HQ + kvh * G + row) * HD + c8;
        v = *reinterpret_cast<const uint4*>(q + off);
      }
      *reinterpret_cast<uint4*>(&sQ[pp * 16 * QS + row * QS + c8]) = v;
    }
    if (tid < 16) { sM[pp * 16 + tid] = -1e30f; sL[pp * 16 + tid] = 0.f; sA[pp * 16 + tid] = 1.f; }
  }
#if !KVFP8
  sLUT[tid] = fp4_pair(tid);
#endif
  for (int i = tid; i < 16 * SS; i += THREADS) sP[i] = __float2half(0.f);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[2][2];
  for (int pp = 0; pp < 2; ++pp) {
    wmma::fill_fragment(o[pp][0], 0.f);
    wmma::fill_fragment(o[pp][1], 0.f);
  }
  /* wmma m16n16 accumulator row of element pair i: this is the mapping that
     validated at rel=4e-4 in the 32-token kernel (not lane>>2). */
  const int rbase = (lane & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);

  const int lo = lo0;
  int t0, t1, nst;
  int ip0 = 0, ip1 = 0;
  if (page_list != nullptr) {
    /* bounded read: the partition is a slice of the selected page list, one
       16-token page per stage (the last page is clamped by kv_len).  Split over
       the pages the table actually holds, not its capacity: the caller passes
       the whole buffer width, and the unused tail used to give every part
       stages that could only produce zeros (measured 4x wasted work at 250K). */
    const int nsel = min(npages_list, (kv_len + BN - 1) / BN);
    ip0 = (int)((long)nsel * sp / nparts);
    ip1 = (int)((long)nsel * (sp + 1) / nparts);
    nst = ip1 - ip0;
    t0 = 0;
    t1 = kv_len;
  } else {
    const int span = span0;
    t0 = lo + (int)((long)span * sp / np);
    t1 = lo + (int)((long)span * (sp + 1) / np);
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
  // sLUT is written by all 256 threads but commit() reads entries written by
  // other warps; without this barrier the first block can dequant with an
  // uninitialised LUT (observed as intermittent non-finite logits under
  // concurrency, 2026-11-03).
  __syncthreads();
  commit(0);
  __syncthreads();

  for (int st = 0; st < nst; ++st) {
    if (st + 1 < nst) fetch(st + 1);   /* LDGs overlap QK/softmax/PV */

    for (int pp = 0; pp < 2; ++pp) {
    /* QK: 4 warps, each 4 of the 16 k-steps -> partial score tiles */
    if (warp < 4) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> sa;
      wmma::fill_fragment(sa, 0.f);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int kk = warp * 4 + k;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
        wmma::load_matrix_sync(qf, &sQ[pp * 16 * QS + kk * 16], QS);
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
      /* t1 is this partition's end, not the request's kv_len: only the
         final partition needs the per-position shortening. */
      const int t1p = (page_list != nullptr)
                          ? min(kv_len - 1 + pp, kv_len)
                          : min(t1, kv_len - 1 + pp);
      if ((page_list != nullptr) ? (gtok_t < t1p) : (at < t1p)) {
        sc = (sSp[head * 16 + tl] + sSp[256 + head * 16 + tl] +
              sSp[512 + head * 16 + tl] + sSp[768 + head * 16 + tl]) * sm_scale;
      }
      float mx = sc;
#pragma unroll
      for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      const float m_old = sM[pp * 16 + head];
      const float m_new = fmaxf(m_old, mx);
      const float p = __expf(sc - m_new);
      float lsum = p;
#pragma unroll
      for (int off = 8; off; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
      sP[(pp * 16 + head) * SS + tl] = __float2half(p);
      if (tl == 0) {   /* running stats are per head, so only lane 0 needs alpha */
        const float alpha = (m_old <= -1e29f) ? 0.f : __expf(m_old - m_new);
        sM[pp * 16 + head] = m_new;
        sL[pp * 16 + head] = sL[pp * 16 + head] * alpha + lsum;
        sA[pp * 16 + head] = alpha;
      }
    }
    __syncthreads();

    /* PV: 8 warps x 2 accumulators (16 dims each), one 16-token k-step */
    {
      const float a0 = sA[pp * 16 + rbase], a2 = sA[pp * 16 + rbase + 2];
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
      wmma::load_matrix_sync(pf, sP + pp * 16 * SS, SS);   /* the same P serves both halves */
#pragma unroll
      for (int t = 0; t < 2; t++) {
        float* acc = o[pp][t].x;
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] *= ((j & 2) ? a2 : a0);
        const int nd = warp * 2 + t;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
        wmma::load_matrix_sync(vf, &sV[nd * 16], KVS);
        wmma::mma_sync(o[pp][t], pf, vf, o[pp][t]);
      }
    }
    __syncthreads();
    }  /* pp */
    if (st + 1 < nst) { commit(st + 1); }
    __syncthreads();
  }

  for (int pp = 0; pp < 2; ++pp) {
    float* scr = reinterpret_cast<float*>(sK);
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 2; t++)
      wmma::store_matrix_sync(scr + (warp * 2 + t) * 256, o[pp][t], 16, wmma::mem_row_major);
    __syncthreads();
    const int rr = r * nqt + q0 + pp;
    for (int i = tid; i < G * HD; i += THREADS) {
      const int row = i / HD, d = i % HD;
      const int wsrc = d / 32, tsrc = (d % 32) / 16, c = d % 16;
      const long rb = (long)(((rr * H + kvh) * G + row) * nparts + sp) * (HD + 2);
      ws[rb + d] = scr[(wsrc * 2 + tsrc) * 256 + row * 16 + c];
    }
    if (tid < G) {
      const long rb = (long)(((rr * H + kvh) * G + tid) * nparts + sp) * (HD + 2);
      ws[rb + HD] = sM[pp * 16 + tid];
      ws[rb + HD + 1] = sL[pp * 16 + tid];
    }
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

  /* How many partitions actually carry work for this request.  Computed here
     and identically in combine_decode, so the two agree on which rows of ws
     are live without any device-side handshake (which would not be safe under
     CUDA-graph capture). */
  const int lo0 = (window >= 0) ? max(kv_len - 1 - window, 0) : 0;
  const int span0 = max(kv_len - lo0, 0);
  const int np = (page_list != nullptr)
                     ? nparts
                     : min(nparts, max(1, span0 / kMinTokensPerPart));
  if (page_list == nullptr && sp >= np) return;

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

  const int lo = lo0;
  int t0, t1, nst;
  int ip0 = 0, ip1 = 0;
  if (page_list != nullptr) {
    /* bounded read: the partition is a slice of the selected page list, one
       16-token page per stage (the last page is clamped by kv_len).  Split over
       the pages the table actually holds, not its capacity: the caller passes
       the whole buffer width, and the unused tail used to give every part
       stages that could only produce zeros (measured 4x wasted work at 250K). */
    const int nsel = min(npages_list, (kv_len + BN - 1) / BN);
    ip0 = (int)((long)nsel * sp / nparts);
    ip1 = (int)((long)nsel * (sp + 1) / nparts);
    nst = ip1 - ip0;
    t0 = 0;
    t1 = kv_len;
  } else {
    const int span = span0;
    t0 = lo + (int)((long)span * sp / np);
    t1 = lo + (int)((long)span * (sp + 1) / np);
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
  // sLUT is written by all 256 threads but commit() reads entries written by
  // other warps; without this barrier the first block can dequant with an
  // uninitialised LUT (observed as intermittent non-finite logits under
  // concurrency, 2026-11-03).
  __syncthreads();
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

// The partition weights depend only on (r, h, qh), not on the output column, so
// one CTA used to derive the same nparts values nparts*blockDim.x times: every
// thread re-ran __expf(ws[b + HD] - m) and reloaded the max/sum words.  At
// HD=256 and nparts=240 that is ~1e6 redundant __expf per call for 3.8e3 useful
// ones, which is why combine_decode cost as much as the attention kernel itself.
// Hoisting them into shared memory keeps the arithmetic identical: each sw[sp]
// is the same expression, and both accumulators still walk sp in order.
constexpr int kMaxCombineParts = 512;

__global__ void combine_decode(const float* __restrict__ ws,
                               __half* __restrict__ out, int nparts, int q0,
                               int nqt, const int* __restrict__ kvlens,
                               int window, int bounded) {
  int r = blockIdx.x, h = blockIdx.y, qh = blockIdx.z;
  int tid = threadIdx.x;
  /* Must match decode_xqa16 exactly: only partitions [0, np) were written.
     The row stride stays nparts (the buffer layout); only the loop bound
     shrinks. */
  const int kv_len = kvlens[r];
  const int lo_c = (window >= 0) ? max(kv_len - 1 - window, 0) : 0;
  const int span_c = max(kv_len - lo_c, 0);
  const int np = bounded ? nparts
                         : min(nparts, max(1, span_c / kMinTokensPerPart));
  const long row0 = (long)((r * H + h) * G + qh) * nparts;
  float m = -1e30f;
  for (int sp = 0; sp < np; sp++)
    m = fmaxf(m, ws[(row0 + sp) * (HD + 2) + HD]);
  __shared__ float sw[kMaxCombineParts];
  __shared__ float s_denom;
  if (np <= kMaxCombineParts) {
    for (int sp = tid; sp < np; sp += blockDim.x)
      sw[sp] = __expf(ws[(row0 + sp) * (HD + 2) + HD] - m);
    // Barrier before thread 0 reads sw[]: without it the denominator can be
    // built from slots other threads have not written yet.  With one write per
    // thread (blockDim >= nparts) the race is usually benign, which is exactly
    // why it survived a bit-identity run at blockDim 256.
    __syncthreads();
    if (tid == 0) {
      float d = 0.f;
      for (int sp = 0; sp < np; sp++)
        d += sw[sp] * ws[(row0 + sp) * (HD + 2) + HD + 1];
      s_denom = d;
    }
    __syncthreads();
    for (int d = tid; d < HD; d += blockDim.x) {
      float acc = 0.f;
      for (int sp = 0; sp < np; sp++)
        acc += sw[sp] * ws[(row0 + sp) * (HD + 2) + d];
      out[(long)((r * nqt + q0) * HQ + h * G + qh) * HD + d] =
          __float2half(acc / (s_denom > 0 ? s_denom : 1.f));
    }
    return;
  }
  float denom = 0.f;
  for (int sp = 0; sp < np; sp++)
    denom += __expf(ws[(row0 + sp) * (HD + 2) + HD] - m) *
             ws[(row0 + sp) * (HD + 2) + HD + 1];
  for (int d = tid; d < HD; d += blockDim.x) {
    float acc = 0.f;
    for (int sp = 0; sp < np; sp++) {
      long b = (row0 + sp) * (HD + 2);
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
      nparts, (int)q0, (int)nqt, kvlens.data_ptr<int>(), window,
      page_list.has_value() ? 1 : 0);
}

static bool g_attr_pair = false;

void run_pair(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
              torch::Tensor kvlens, torch::Tensor kvlens_rows,
              torch::Tensor ws, torch::Tensor out, double k_scale,
              double v_scale, double sm_scale, int nparts, int window, int R,
              c10::optional<torch::Tensor> page_list,
              c10::optional<torch::Tensor> lut = c10::nullopt,
              int64_t hstride = 0, int64_t tstride = 0, int64_t npg = 0,
              int64_t q0 = 0, int64_t nqt = 1) {
#if KVFP8 || KVFP16
  const int H2 = kv.size(1);
#else
  const int H2 = kv.size(1) / 2;
#endif
  const long page = kv.stride(0);
  const int smem = (2 * 16 * QS + 2 * BN * KVS) * 2 + 4 * 16 * 16 * 4
                   + 2 * 16 * SS * 2 + 2 * 16 * 12 + 1024;
  if (!g_attr_pair) {
    cudaFuncSetAttribute(decode_xqa16_pair,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g_attr_pair = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 g(nparts, H2, R);
  for (int p0 = 0; p0 + 1 < (int)nqt; p0 += 2) {
    torch::Tensor kvl = kvlens + p0;   // this pair's longest KV
    decode_xqa16_pair<<<g, THREADS, smem, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        kv.data_ptr<unsigned char>(), bt.data_ptr<int>(),
        kvl.data_ptr<int>(), ws.data_ptr<float>(), (float)k_scale,
        (float)v_scale, (float)sm_scale, bt.size(1), nparts, window, page,
        page / 2,
        page_list.has_value() ? page_list->data_ptr<int>() : nullptr,
        page_list.has_value() ? (int)page_list->numel() : 0,
        lut.has_value()
            ? reinterpret_cast<const __half*>(lut->data_ptr<at::Half>())
            : nullptr,
        (int)hstride, (int)tstride, (int)npg, (int)q0 + p0, (int)nqt);
  }
  dim3 gc(R * (int)nqt, H2, G);
  combine_decode<<<gc, 256, 0, stream>>>(
      ws.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      nparts, 0, 1, kvlens_rows.data_ptr<int>(), window,
      page_list.has_value() ? 1 : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); m.def("run_pair", &run_pair); m.def("prepare", &prepare); }
"""

_MODS_XQA = {}


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
    # Compile every admitted geometry before CUDA-graph capture: compiling or
    # allocating inside capture breaks the graph, and warmup does not know
    # which TP split the service will use.
    from .sm70_attention_geometry import VERIFIED_GEOMETRIES

    for _h, _hq in sorted(VERIFIED_GEOMETRIES):
        if os.environ.get("VLLM_SM70_DECODE_XQA") == "1":
            _mod_xqa(_h, _hq).prepare()
        _mod(_h, _hq)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _MAXR = max(int(max_reqs), 1)
    _slots = _NPARTS
    # H*RATIO == HQ. 16 covers TP1 (2*8) and TP2 (1*8); TP4 (1*4) over-allocates.
    n = _MAXR * 16 * _slots * 258  # R*H*RATIO*NPARTS*(D+2)
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
        if os.environ.get("VLLM_SM70_XQA_PAIR") == "1" and nqt % 2 == 0:
            from vllm.v1.attention.ops.sm70_nvfp4_dequant import _fp8_lut

            R_, HQ_, D_ = q.shape[0] // nqt, q.shape[1], q.shape[2]
            H_ = int(kv_cache.shape[1])
            kv_u8_ = (kv_cache if kv_cache.dtype == torch.uint8
                      else kv_cache.view(torch.uint8))
            lut_ = _fp8_lut(kv_u8_.device, kv_dtype)
            rows = (base[:, None]
                    + torch.arange(nqt, device=base.device)[None, :]
                    ).reshape(-1).to(torch.int32)
            mod = _mod_xqa_fp8(H_, HQ_)
            mod.prepare()
            mod.run_pair(q.contiguous(), kv_u8_, block_table, base, rows,
                         workspace, out, float(k_scale), float(v_scale),
                         float(sm_scale), int(splits), int(window), R_, None,
                         lut_, kv_u8_.stride(1), kv_u8_.stride(2),
                         int(kv_u8_.size(2)), 0, nqt)
            return
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
    # The pybind binding does not expose C++ default arguments, so every
    # parameter must be passed: q0 and nqt close the 19-argument signature
    # (the fp16 caller already does this).
    mod.run(q.contiguous(), kv_u8, block_table, kv_lens, workspace, out,
            float(k_scale), float(v_scale), float(sm_scale), int(splits),
            int(window), R, None, lut, kv_u8.stride(1), kv_u8.stride(2),
            int(kv_u8.size(2)), 0, 1)


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
    # Verified geometries: TP1 (H=2, HQ=16), TP2 (H=1, HQ=8).
    if not supported_decode(H, HQ, D):
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
        nparts = _NPARTS_XQA
        if os.environ.get("VLLM_SM70_DECODE_XQA_RSCALE") == "1":
            # Batch-scaled parts: ws write traffic (R*HQ*np*258 fp32) and the
            # combine re-read grow with np, while total CTAs (np*H2*R) only
            # need ~2 waves to hide the bt->kv dependent-load chain. Measured
            # 2026-11-06 at the true conc shapes (screens/xqa_true_shape.py,
            # CUPTI min-of-3): R=8 kv20K best np=30 (361.6us vs 498.7 at 240),
            # R=16 kv20K best np=15 (697.7 vs 1115.4), R=1 kv65K best np=120
            # (187.3 vs 220.4); below ~300 CTAs the kernel collapses to
            # latency-bound (R=16 np=8: 1022us). round(240/R) hits every
            # measured optimum; clamp [15,120] keeps low-R parallelism.
            nparts = max(15, min(120, (240 + R // 2) // max(R, 1)))
        need = R * HQ * nparts * 258
        _mod_xqa(H, HQ).run(q, kv_cache, block_table, kv_lens, _BUF[:need], out,
                            k_scale, v_scale, sm_scale, nparts,
                            int(window_left), R, None, None, 0, 0,
                            int(kv_cache.size(2)), int(q0), int(nqt))
        return True
    if nqt != 1 or q0 != 0:
        return False  # spec-decode verify is XQA-only; use the Triton oracle
    need = R * HQ * _NPARTS * 258
    # The scalar fallback (VLLM_SM70_DECODE_XQA=0) takes the compact signature;
    # it has no page-list / q0 / nqt arguments.
    _mod(H, HQ).run(q, kv_cache, block_table, kv_lens, _BUF[:need], out,
                    k_scale, v_scale, sm_scale, _NPARTS, int(window_left), R)
    return True
