#!/usr/bin/env python3
"""Decode-shape microbenchmark for the SM70 FP8 QPN GEMM (the #1 decode kernel).

At TP1 batch=1 this kernel is 2.72 ms of the 10.64 ms decode step (25.5%), called
81 times per step at ~34 us.  The launcher derives its K-split from
``target = 5120/split_k`` but then clamps it with ``VLLM_SM70_FP8_QPN_KSPLIT``
whose default is 1, so the split-K path never runs in production and small-n
shapes launch only n/32 CTAs (64 CTAs for o_proj on an 80-SM GPU).

This measures, per production shape, the achieved bandwidth against the 900 GB/s
HBM ceiling for each K-split, with cuBLAS on a dequantised FP16 copy as the
honest comparator.  The env var here is a *measurement arm*, not a deployment
knob: the deliverable is a correct launcher heuristic.

Timing is CUDA-graph replay (reps launches captured back to back, min over
attempts), because CUPTI is not installed in this environment and the
event-based fallback under flashinfer reports zeros.  Bytes counted are the
weights read once; every shape here exceeds the 6 MB L2, so the reads are cold.

    for cap in 1 2 4 8; do VLLM_SM70_FP8_QPN_KSPLIT=$cap \\
        python sm70_tests/bench_fp8_qpn.py; done
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# (name, n, k, calls_per_step) — derived from config.json and the trace grid dims
# grid=(384,1,1) -> n=12288 : GDN in_proj (2*2048 key + 2*4096 value)
# grid=(288,1,1) -> n= 9216 : full-attn qkv+gate (2*4096 + 2*512)
# grid=( 64,1,1) -> n= 2048, k=4096 : o_proj / GDN out_proj
# grid=(7760,1,1)-> n=248320 : lm_head (vocab)
SHAPES = [
    ("gdn_in_proj", 12288, 2048, 30),
    ("attn_qkv", 9216, 2048, 10),
    ("o_proj", 2048, 4096, 40),
    ("lm_head", 248320, 2048, 1),
]
HBM_GBPS = 900.0


def graph_time(fn, reps: int, attempts: int = 7) -> float:
    """us per call: capture ``reps`` launches in a CUDA graph, replay, take the
    min over attempts (same-process interleaving, min — clock drift across
    processes is why absolute numbers from different runs are not comparable)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    best = float("inf")
    for _ in range(attempts):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        g.replay()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en))
    return best * 1e3 / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--split-k", type=int, default=16, choices=[8, 16])
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cap = os.environ.get("VLLM_SM70_FP8_QPN_KSPLIT", "unset(=1)")
    import vllm  # noqa: F401
    # the _C extension loads lazily via the platform; without this torch.ops._C
    # has no members
    import vllm._custom_ops  # noqa: F401

    dev = "cuda"
    print(f"# m={args.m} split_k={args.split_k} "
          f"VLLM_SM70_FP8_QPN_KSPLIT={cap}  gpu={torch.cuda.get_device_name(0)}")
    print(f"{'shape':>12} {'n':>7} {'k':>5} {'MB':>7} {'qpn_us':>8} {'GB/s':>7}"
          f" {'%HBM':>6} {'cublas_us':>9} {'GB/s':>7} {'q/c':>6} {'rel':>9}")
    rows = []
    for name, n, k, _ in SHAPES:
        torch.manual_seed(9502)
        # valid e4m3 codes: quantise real values, never random bytes (0x7F is NaN)
        w = torch.randn(k, n, device=dev, dtype=torch.float32).clamp_(-4.0, 4.0)
        w8 = w.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        w_ref = w8.view(torch.float8_e4m3fn).float()
        packed = torch.ops._C.sm70_fp8_qpn_pack(w8)
        scale = torch.ones(1, dtype=torch.float32, device=dev)
        a = torch.randn(args.m, k, dtype=torch.float16, device=dev)
        out = torch.empty(args.m, n, dtype=torch.float16, device=dev)

        def qpn(a=a, out=out, packed=packed, scale=scale):
            torch.ops._C.sm70_fp8_qpn_dispatch(out, a, packed, scale,
                                               args.split_k)

        w16 = torch.ops._C.sm70_fp8_qpn_dequant(packed, scale, k, n)

        def cublas(a=a, w16=w16):
            return a @ w16

        qpn()
        torch.cuda.synchronize()
        ref = a.float() @ w_ref
        rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()

        t_q = graph_time(qpn, args.reps)
        t_c = graph_time(cublas, args.reps)
        mb = n * k / 1e6
        gq, gc = mb / 1e3 / (t_q / 1e6), mb / 1e3 / (t_c / 1e6)
        print(f"{name:>12} {n:>7} {k:>5} {mb:>7.1f} {t_q:>8.1f} {gq:>7.0f}"
              f" {gq/HBM_GBPS*100:>5.1f}% {t_c:>9.1f} {gc:>7.0f} {t_c/t_q:>6.2f}"
              f" {rel:>9.1e}")
        rows.append(dict(shape=name, n=n, k=k, m=args.m, cap=str(cap),
                         split_k=args.split_k, us_qpn=t_q, us_cublas=t_c,
                         gbps_qpn=gq, gbps_cublas=gc, rel=rel))
        del w, w8, w_ref, packed, w16, a, out
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
