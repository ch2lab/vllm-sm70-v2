# SPDX-License-Identifier: Apache-2.0
"""Hand-written SM70 NVFP4 decode kernel (scalar FMA, split-KV) as a faster
alternative to the Triton sm70_nvfp4_paged_decode. Env-gated (default off);
shape-gated to (D=256,H=2,HQ=16,N=16); falls back to Triton otherwise.
CG-safe: reuses the caller-provided float workspace for split partials."""
import os
import torch
from torch.utils.cpp_extension import load_inline

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

// D=256, H=2, HQ=16, RATIO=8, N=16, DD=128, SD=16, F=144, page=9216
#define D 256
#define H 2
#define HQ 16
#define RATIO 8
#define N 16
#define DD 128
#define SD 16
#define F 144
#define PAGE 9216

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


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(name="sm70_nvfp4_decode_cuda_v2", cpp_sources="",
                           cuda_sources=_CUDA, verbose=False,
                           extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _MOD


def warmup_compile(device=None, max_reqs=8):
    """Compile the module AND allocate the persistent 240-part scratch buffer,
    both BEFORE CUDA-graph capture (compiling/allocating inside capture breaks
    the graph)."""
    global _BUF, _MAXR
    _mod()
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _MAXR = max(int(max_reqs), 1)
    n = _MAXR * 2 * 8 * _NPARTS * 258  # R*H*RATIO*NPARTS*(D+2)
    if _BUF is None or _BUF.numel() < n or _BUF.device != device:
        _BUF = torch.zeros(n, dtype=torch.float32, device=device)


def try_paged_decode_cuda(q, kv_cache, block_table, kv_lens, out,
                          k_scale, v_scale, sm_scale, window_left, splits, workspace):
    if os.environ.get("VLLM_SM70_DECODE_CUDA") != "1":
        return False
    if q.dtype != torch.float16 or q.dim() != 3:
        return False
    R, HQ, D = q.shape
    if D != 256 or HQ != 16 or kv_cache.shape[1] != 4:
        return False
    if _BUF is None or R > _MAXR:
        return False
    q = q.contiguous()
    if not out.is_contiguous():
        return False
    need = R * 2 * 8 * _NPARTS * 258
    _mod().run(q, kv_cache, block_table, kv_lens, _BUF[:need], out,
               k_scale, v_scale, sm_scale, _NPARTS, int(window_left), R)
    return True
