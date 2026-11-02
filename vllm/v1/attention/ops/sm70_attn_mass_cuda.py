# SPDX-License-Identifier: Apache-2.0
"""Per-(layer, q head) attention-mass probe for the SM70 nvfp4 KV cache.

Profiling only, used by ``sm70_tests/profile_streaming_heads.py``.  For one
decode position it computes the exact softmax over the *whole* cache and reports
what share of the probability mass lands on the sink tokens and on the recent
window.  Heads whose mass is almost entirely there are the "streaming" heads of
LServe / DuoAttention: they do not need content-based page selection, only the
sink and the window.

The probe never touches the attention output, so enabling it cannot change a
result; it is skipped while a CUDA graph is being captured, which keeps it out
of the captured graph.

Valve: ``VLLM_SM70_PROBE_MASS=<csv path>`` (default unset = off).  Probe on a
run with ``--enforce-eager`` so the Python branch actually executes.
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float e2m1_p(int n) {
  const float L[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = L[n & 7];
  return (n & 8) ? -v : v;
}

__device__ __forceinline__ float e4m3_p(unsigned char b) {
  int s = (b & 0x80) ? -1 : 1;
  int e = (b >> 3) & 0xF, m = b & 7;
  float v = (e == 0) ? (m * 0.001953125f) : ((1.f + m * 0.125f) * exp2f((float)(e - 7)));
  return s * v;
}

// One CTA per q head; each warp walks a strided token subset and keeps an online
// (max, sum) softmax with separate sink / window accumulators.  A warp owns one
// token at a time: 32 lanes x 8 dims = the 256-dim head.
__global__ void mass_probe_kernel(const __half* __restrict__ q,
                                  const unsigned char* __restrict__ kv,
                                  const int* __restrict__ bt,
                                  const int* __restrict__ kvlens,
                                  float* __restrict__ out, int HQ, int D, int H,
                                  int NP, int DD, int SD, int page_bytes,
                                  int kv_len, int sink_tok, int win_tok, int G,
                                  float sm) {
  const int hq = blockIdx.x;
  const int h = hq / G;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;
  const __half* qp = q + (long)hq * D;
  const int w0 = kv_len - win_tok;
  float m = -1e30f, l = 0.f, ls = 0.f, lw = 0.f;
  for (int t = warp; t < kv_len; t += nwarp) {
    const int pg = t / NP, tt = t % NP;
    const unsigned char* base = kv + (long)bt[pg] * page_bytes;
    const unsigned char* kd = base + (long)h * (NP * DD) + (long)tt * DD;
    const unsigned char* ks = base + (long)H * (NP * DD) + (long)h * (NP * SD) +
                              (long)tt * SD;
    const float sc = e4m3_p(ks[lane >> 1]);
    const unsigned int w =
        *reinterpret_cast<const unsigned int*>(kd + lane * 4);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned char byte = (w >> (j * 8)) & 0xFFu;
      acc += __half2float(qp[lane * 8 + j * 2]) * e2m1_p(byte & 0xF) * sc;
      acc += __half2float(qp[lane * 8 + j * 2 + 1]) * e2m1_p(byte >> 4) * sc;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    const float x = acc * sm;
    float scale = 1.f;
    if (x > m) {
      scale = __expf(m - x);
      m = x;
    }
    const float e = __expf(x - m);
    l = l * scale + e;
    ls *= scale;
    lw *= scale;
    if (t < sink_tok) ls += e;
    if (t >= w0) lw += e;
  }
  __shared__ float r_m[32], r_l[32], r_s[32], r_w[32];
  if (lane == 0) {
    r_m[warp] = m;
    r_l[warp] = l;
    r_s[warp] = ls;
    r_w[warp] = lw;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mm = -1e30f;
    for (int i = 0; i < nwarp; ++i) mm = fmaxf(mm, r_m[i]);
    float L = 0.f, LS = 0.f, LW = 0.f;
    for (int i = 0; i < nwarp; ++i) {
      const float f = __expf(r_m[i] - mm);
      L += r_l[i] * f;
      LS += r_s[i] * f;
      LW += r_w[i] * f;
    }
    out[hq * 2 + 0] = LS / L;
    out[hq * 2 + 1] = LW / L;
  }
}

void mass_probe(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
                torch::Tensor kvlens, torch::Tensor out, int64_t sink_tok,
                int64_t win_tok, double sm) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = kv.size(1) / 2;
  const int NP = kv.size(2);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  const int kv_len = kvlens[0].item<int>();
  mass_probe_kernel<<<HQ, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      out.data_ptr<float>(), HQ, D, H, NP, DD, SD, page_bytes, kv_len,
      (int)sink_tok, (int)win_tok, G, (float)sm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mass_probe", &mass_probe); }
"""

_MOD = None
_STATE = {"order": {}, "rows": 0, "path": None, "budget": 20000}


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(name="sm70_attn_mass", cpp_sources="",
                           cuda_sources=_CUDA, verbose=False,
                           extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _MOD


def enabled() -> bool:
    return bool(os.environ.get("VLLM_SM70_PROBE_MASS"))


def probe(qo, q, kv, block_table, kv_lens, sm_scale, sink_tok, win_tok) -> None:
    """Append one CSV row per (request, layer, q head) for this prefill call."""
    if not enabled() or torch.cuda.is_current_stream_capturing():
        return
    st = _STATE
    if st["rows"] >= st["budget"]:
        return
    if st["path"] is None:
        st["path"] = os.environ["VLLM_SM70_PROBE_MASS"]
        with open(st["path"], "w") as fh:
            fh.write("layer,qhead,kv_len,sink_frac,win_frac\n")
    ptr = kv.data_ptr()
    layer = st["order"].get(ptr)
    HQ = int(q.shape[1])
    out = torch.empty(2 * HQ, dtype=torch.float32, device=q.device)
    for r in range(int(block_table.shape[0])):
        q1 = int(qo[r + 1])
        if q1 <= int(qo[r]):
            continue
        kvl = kv_lens[r:r + 1]
        klen = int(kvl[0])
        if klen < 4096:
            # Startup warmup runs a dummy batch; recording it would also poison
            # the layer-order map with pointers that never appear again.
            continue
        if layer is None:
            layer = st["order"].setdefault(ptr, len(st["order"]))
        _mod().mass_probe(q[q1 - 1].contiguous(), kv, block_table[r], kvl, out,
                          sink_tok, win_tok, sm_scale)
        vals = out.tolist()
        with open(st["path"], "a") as fh:
            for h in range(HQ):
                fh.write(f"{layer},{h},{klen},{vals[2 * h]:.6f},"
                         f"{vals[2 * h + 1]:.6f}\n")
        st["rows"] += HQ
