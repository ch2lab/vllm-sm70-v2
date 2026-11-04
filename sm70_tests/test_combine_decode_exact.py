#!/usr/bin/env python3
"""Prove combine_decode's shared-memory hoist is bit-identical, and price it.

combine_decode reduced nparts fp32 partials per output column, but every thread
recomputed the per-partition softmax weight __expf(ws[b+HD]-m) and reloaded the
max/sum words, so one CTA derived the same nparts values nparts*blockDim.x times.
The fix hoists those into shared memory.  That is only free if the arithmetic is
unchanged, so this compiles the old and new bodies side by side and compares
their outputs bit for bit, then prices the difference with CUDA-graph replay.

    python sm70_tests/test_combine_decode_exact.py
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

# mirror of the device-side constants below, for the host-side index maths
H, HQ = 2, 16
G, HD = HQ // H, 256

HDR = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
constexpr int H = 2, HQ = 16, G = HQ / H, HD = 256;
"""

# the body exactly as it was before the hoist
OLD = r"""
__global__ void combine_ref(const float* __restrict__ ws, __half* __restrict__ out,
                            int nparts, int q0, int nqt) {
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
"""

NEW = r"""
constexpr int kMaxCombineParts = 512;
__global__ void combine_ds(const float* __restrict__ ws, __half* __restrict__ out,
                           int nparts, int q0, int nqt, int ds) {
  int r = blockIdx.x, h = blockIdx.y;
  int qh = blockIdx.z / ds, sub = blockIdx.z % ds;
  int tid = threadIdx.x, nth = blockDim.x;
  const long row0 = (long)((r * H + h) * G + qh) * nparts;
  float m = -1e30f;
  for (int sp = 0; sp < nparts; sp++)
    m = fmaxf(m, ws[(row0 + sp) * (HD + 2) + HD]);
  __shared__ float sw[kMaxCombineParts];
  __shared__ float s_denom;
  if (nparts > kMaxCombineParts) {                    // cannot stage the weights
    for (int d = sub * nth + tid; d < HD; d += nth * ds) {
      float acc = 0.f, denom = 0.f;
      for (int sp = 0; sp < nparts; sp++)
        denom += __expf(ws[(row0 + sp) * (HD + 2) + HD] - m) *
                 ws[(row0 + sp) * (HD + 2) + HD + 1];
      for (int sp = 0; sp < nparts; sp++) {
        long bb = (row0 + sp) * (HD + 2);
        acc += __expf(ws[bb + HD] - m) * ws[bb + d];
      }
      out[(long)((r * nqt + q0) * HQ + h * G + qh) * HD + d] =
          __float2half(acc / (denom > 0 ? denom : 1.f));
    }
    return;
  }
  for (int sp = tid; sp < nparts; sp += nth)
    sw[sp] = __expf(ws[(row0 + sp) * (HD + 2) + HD] - m);
  // Barrier before thread 0 reads sw[]: without it the denominator can be built
  // from slots other threads have not written yet.  With one write per thread
  // (blockDim >= nparts) the race is usually benign, which is exactly why it
  // survived the first bit-identity run; at blockDim 64 / nparts 240 it is not.
  __syncthreads();
  if (tid == 0) {
    float d = 0.f;
    for (int sp = 0; sp < nparts; sp++)
      d += sw[sp] * ws[(row0 + sp) * (HD + 2) + HD + 1];
    s_denom = d;
  }
  __syncthreads();
  for (int d = sub * nth + tid; d < HD; d += nth * ds) {
    float acc = 0.f;
    for (int sp = 0; sp < nparts; sp++)
      acc += sw[sp] * ws[(row0 + sp) * (HD + 2) + d];
    out[(long)((r * nqt + q0) * HQ + h * G + qh) * HD + d] =
        __float2half(acc / (s_denom > 0 ? s_denom : 1.f));
  }
}

__global__ void combine_new(const float* __restrict__ ws, __half* __restrict__ out,
                            int nparts, int q0, int nqt) {
  int r = blockIdx.x, h = blockIdx.y, qh = blockIdx.z;
  int tid = threadIdx.x;
  const long row0 = (long)((r * H + h) * G + qh) * nparts;
  float m = -1e30f;
  for (int sp = 0; sp < nparts; sp++)
    m = fmaxf(m, ws[(row0 + sp) * (HD + 2) + HD]);
  __shared__ float sw[kMaxCombineParts];
  __shared__ float s_denom;
  if (nparts <= kMaxCombineParts) {
    for (int sp = tid; sp < nparts; sp += blockDim.x)
      sw[sp] = __expf(ws[(row0 + sp) * (HD + 2) + HD] - m);
    if (tid == 0) {
      float d = 0.f;
      for (int sp = 0; sp < nparts; sp++)
        d += sw[sp] * ws[(row0 + sp) * (HD + 2) + HD + 1];
      s_denom = d;
    }
    __syncthreads();
    for (int d = tid; d < HD; d += blockDim.x) {
      float acc = 0.f;
      for (int sp = 0; sp < nparts; sp++)
        acc += sw[sp] * ws[(row0 + sp) * (HD + 2) + d];
      out[(long)((r * nqt + q0) * HQ + h * G + qh) * HD + d] =
          __float2half(acc / (s_denom > 0 ? s_denom : 1.f));
    }
    return;
  }
  float denom = 0.f;
  for (int sp = 0; sp < nparts; sp++)
    denom += __expf(ws[(row0 + sp) * (HD + 2) + HD] - m) *
             ws[(row0 + sp) * (HD + 2) + HD + 1];
  for (int d = tid; d < HD; d += blockDim.x) {
    float acc = 0.f;
    for (int sp = 0; sp < nparts; sp++) {
      long b = (row0 + sp) * (HD + 2);
      acc += __expf(ws[b + HD] - m) * ws[b + d];
    }
    out[(long)((r * nqt + q0) * HQ + h * G + qh) * HD + d] =
        __float2half(acc / (denom > 0 ? denom : 1.f));
  }
}
"""

HOST = r"""
void run_ref(torch::Tensor ws, torch::Tensor out, int R, int nparts,
             int q0, int nqt) {
  dim3 g(R, H, G);
  combine_ref<<<g, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      ws.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), nparts, q0, nqt);
}
void run_new(torch::Tensor ws, torch::Tensor out, int R, int nparts,
             int q0, int nqt) {
  dim3 g(R, H, G);
  combine_new<<<g, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      ws.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), nparts, q0, nqt);
}
// same reduction, but the HD columns are spread over ds CTAs so more SMs take
// part; each output element is still produced by the same ordered loop over sp
void run_ds(torch::Tensor ws, torch::Tensor out, int R, int nparts,
            int q0, int nqt, int64_t ds) {
  dim3 g(R, H, (unsigned)(G * ds));
  combine_ds<<<g, (unsigned)(HD / ds), 0, c10::cuda::getCurrentCUDAStream()>>>(
      ws.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), nparts, q0, nqt,
      (int)ds);
}
"""

SRC = HDR + OLD + NEW + HOST
CPP = ("void run_ref(torch::Tensor ws, torch::Tensor out, int R, int nparts,"
       " int q0, int nqt);\n"
       "void run_new(torch::Tensor ws, torch::Tensor out, int R, int nparts,"
       " int q0, int nqt);\n"
       "void run_ds(torch::Tensor ws, torch::Tensor out, int R, int nparts,"
       " int q0, int nqt, int64_t ds);")


def graph_time(fn, reps=100, attempts=7):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    best = float("inf")
    for _ in range(attempts):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best * 1e3 / reps


def main() -> int:
    m = load_inline(name="combine_exact_test", cpp_sources=CPP,
                    cuda_sources=SRC,
                    functions=["run_ref", "run_new", "run_ds"],
                    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    bad = 0
    for R in (1, 2, 8):
        for nparts in (120, 240, 512, 600):
            torch.manual_seed(9502)
            n_rows = R * H * G * nparts
            ws = torch.randn(n_rows, 258, device="cuda", dtype=torch.float32)
            ws[:, 256] *= 4.0                      # per-partition max
            ws[:, 257] = ws[:, 257].abs() * 8.0    # per-partition sum(exp), > 0
            a = torch.zeros(R * 16 * 256, device="cuda", dtype=torch.float16)
            b = torch.zeros_like(a)
            m.run_ref(ws, a, R, nparts, 0, 1)
            m.run_new(ws, b, R, nparts, 0, 1)
            torch.cuda.synchronize()
            same = torch.equal(a.view(torch.int16), b.view(torch.int16))
            finite = bool(torch.isfinite(b.float()).all())
            t_ref = graph_time(lambda: m.run_ref(ws, a, R, nparts, 0, 1))
            t_new = graph_time(lambda: m.run_new(ws, b, R, nparts, 0, 1))
            mb = n_rows * 258 * 4 / 1e6          # fp32 partials read
            print(f"R={R:2d} nparts={nparts:4d} bit-identical={str(same):>5} "
                  f"finite={str(finite):>5}  ref={t_ref:7.2f}us new={t_new:7.2f}us "
                  f"({t_ref/t_new:5.2f}x, {mb/1e3/(t_new/1e6):5.0f} GB/s)")
            if not (same and finite):
                bad += 1
            for ds in (2, 4, 8):
                c = torch.zeros_like(a)
                m.run_ds(ws, c, R, nparts, 0, 1, ds)
                torch.cuda.synchronize()
                sd = torch.equal(a.view(torch.int16), c.view(torch.int16))
                td = graph_time(
                    lambda: m.run_ds(ws, c, R, nparts, 0, 1, ds))
                print(f"                 ds={ds}: bit-identical={str(sd):>5} "
                      f"{td:7.2f}us ({t_ref/td:5.2f}x vs ref, "
                      f"{mb/1e3/(td/1e6):5.0f} GB/s, "
                      f"{R*H*G*ds:3d} CTAs)")
                if not sd:
                    bad += 1
    print("RESULT:", "all bit-identical" if bad == 0 else f"{bad} MISMATCHES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
