#!/usr/bin/env python3
"""Why does _sm70_topk_softmax_kernel cost 10.2 us in the service but 4.2 us
in bench_router_topk.py?  Two candidates, measured against each other:

  HAS_PAD  - decode runs under a CUDA graph, so every row carries an
             is_padding tensor and the kernel takes the padded branch.
  warps    - _topk_warps(rows) returns 4 below 256 rows, but at rows=1 the
             whole kernel is one CTA, so extra warps only add cross-warp
             reductions around each of the K serial argmax passes.

Also sweeps K so the fixed launch cost can be separated from the per-argmax
marginal cost: if the intercept is most of the 10 us, no amount of argmax
cleverness will pay, and this line should be closed rather than rewritten.

    CUDA_VISIBLE_DEVICES=1 python sm70_tests/probe_topk_cost.py
"""
from __future__ import annotations

import torch
import triton

from vllm.sm70_topk import _sm70_topk_softmax_kernel

B, E = 1, 256


def timed(fn, reps: int = 50, attempts: int = 9) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    best = float("inf")
    for _ in range(attempts):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b))
    return best * 1e3 / reps


def main() -> int:
    dev = "cuda"
    gate = torch.randn(B, E, device=dev, dtype=torch.bfloat16)
    pad = torch.zeros(B, device=dev, dtype=torch.int32)
    print(f"# b={B} E={E} gpu={torch.cuda.get_device_name(0)}")
    print(f"{'K':>3} {'HAS_PAD':>8} {'warps':>6} {'us/call':>9}")
    base = {}
    for K in (1, 2, 4, 8):
        k2 = triton.next_power_of_2(K)
        wk = torch.zeros(B, K, device=dev, dtype=torch.float32)
        ik = torch.zeros(B, K, device=dev, dtype=torch.int32)
        tk = torch.zeros(B, K, device=dev, dtype=torch.int32)
        for has_pad in (False, True):
            for nw in (1, 2, 4):
                def call(K=K, k2=k2, nw=nw, has_pad=has_pad, wk=wk, ik=ik, tk=tk):
                    _sm70_topk_softmax_kernel[(B,)](
                        gate, wk, ik, tk, pad,
                        gate.stride(0), wk.stride(0), ik.stride(0), tk.stride(0),
                        renormalize=True, HAS_PAD=has_pad, E=E, K=K, K2=k2,
                        BLOCK_E=max(triton.next_power_of_2(E), 2 * K),
                        num_warps=nw)
                t = timed(call)
                base[(K, has_pad, nw)] = t
                print(f"{K:>3} {str(has_pad):>8} {nw:>6} {t:>9.2f}")
    print("\n# fixed-vs-marginal: linear fit of us/call over K (warps=4, HAS_PAD=True)")
    pts = [(K, base[(K, True, 4)]) for K in (1, 2, 4, 8)]
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    icept = (sy - slope * sx) / n
    print(f"  intercept (launch+teardown) = {icept:.2f} us")
    print(f"  per extra argmax pass       = {slope:.2f} us")
    print(f"  K=8 predicted               = {icept + 8 * slope:.2f} us")
    print("  => if the intercept dominates, an argmax rewrite cannot pay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
