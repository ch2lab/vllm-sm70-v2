#!/usr/bin/env python3
"""Price the decode router kernel and check whether its warp count is tunable
without changing the selection.

At decode (b=1) _sm70_topk_softmax_kernel runs as ONE CTA that performs K=8
sequential 256-element max/argmax reductions; the profile says 9.8 us per call,
40 calls per step (0.392 ms/step) for about 1 KB of real work -- i.e. it is
launch/occupancy bound, not work bound.

max/argmax are order-independent, so num_warps cannot change which experts are
selected; tl.sum for the softmax denominator is order-dependent, so the weights
can move by an ULP.  This measures both: timing per warp count, and whether ids
and weights actually differ.
"""
from __future__ import annotations

import sys

import torch
import triton

sys.path.insert(0, "/data/src/vllm-v2")
from vllm.sm70_topk import _sm70_topk_softmax_kernel  # noqa: E402

B, E, K = 1, 256, 8


def graph_time(fn, reps=200, attempts=7):
    for _ in range(5):
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


def run(gate, w, idx, tei, nw):
    _sm70_topk_softmax_kernel[(B,)](
        gate, w, idx, tei, idx, gate.stride(0), w.stride(0), idx.stride(0),
        tei.stride(0), renormalize=True, HAS_PAD=False, E=E, K=K,
        K2=triton.next_power_of_2(K), BLOCK_E=max(triton.next_power_of_2(E), 2 * K),
        num_warps=nw)


def main() -> int:
    torch.manual_seed(9502)
    gate = torch.randn(B, E, device="cuda", dtype=torch.float32) * 2.0
    ref_w = ref_i = None
    print(f"# b={B} E={E} K={K} gpu={torch.cuda.get_device_name(0)}")
    print(f"{'warps':>6} {'us/call':>9} {'ids==ref':>9} {'w==ref':>7} "
          f"{'max|dw|':>10}")
    for nw in (1, 2, 4, 8):
        w = torch.zeros(B, K, device="cuda", dtype=torch.float32)
        idx = torch.zeros(B, K, device="cuda", dtype=torch.int32)
        tei = torch.zeros(B, K, device="cuda", dtype=torch.int32)
        run(gate, w, idx, tei, nw)
        torch.cuda.synchronize()
        if nw == 1:
            ref_w, ref_i = w.clone(), idx.clone()
        t = graph_time(lambda: run(gate, w, idx, tei, nw))
        same_i = bool(torch.equal(idx, ref_i))
        same_w = bool(torch.equal(w, ref_w))
        dw = float((w - ref_w).abs().max())
        print(f"{nw:>6} {t:>9.2f} {str(same_i):>9} {str(same_w):>7} {dw:>10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
