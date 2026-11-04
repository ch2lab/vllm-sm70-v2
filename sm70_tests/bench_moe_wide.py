#!/usr/bin/env python3
"""Reproduce the prefill MoE expert GEMM offline and price it against the
123 TFLOPS HMMA ceiling (the kernel is 17.03 s / 11.8% of a 250K prefill at
only 24% of peak, per the 2026-11-05 attribution).

Shapes are the real ones: a full 8192-token chunk, top_k=8 over 256 experts,
w13 n=1024 k=2048 and w2 n=2048 k=512, route blocks of 16.  Routing is
synthetic but uniform (256 rows per expert, so no padding waste), which is what
the kernel's cost depends on; correctness of any change is proven separately by
an A/B bit-identity test, not by comparing against a reference GEMM here.

The env vars below are MEASUREMENT ARMS for the launcher's kernel selection,
mirroring what serve_qwen36.sh sets in production.  A winning arm becomes a
code default, not a knob the operator must set.

    VLLM_SM70_MOE_QPN2_WIDE=256 VLLM_SM70_MOE_QPN2_WARPS=16 \\
    VLLM_SM70_MOE_QPN2_NS=1 python sm70_tests/bench_moe_wide.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

TOPK = 8
E = 256
BLOCK = 16
PEAK_TFLOPS = 123.0


def graph_time(fn, reps=20, attempts=5):
    for _ in range(2):
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


def build(m, n, k, shards, dev):
    torch.manual_seed(9502)
    codes = torch.randint(0, 255, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scales = torch.randint(1, 120, (E, n, k // 16), dtype=torch.uint8,
                           device=dev)
    gscale = torch.rand((E, shards) if shards > 1 else (E,),
                        dtype=torch.float32, device=dev) * 0.01 + 0.001
    inp = torch.randn(m, k, dtype=torch.float16, device=dev)
    out = torch.zeros(m * TOPK, n, dtype=torch.float16, device=dev)
    per_e = m * TOPK // E
    assert per_e % BLOCK == 0, per_e
    blocks = E * per_e // BLOCK
    st = torch.empty(blocks * BLOCK, dtype=torch.int32, device=dev)
    eid = torch.empty(blocks, dtype=torch.int32, device=dev)
    for b in range(blocks):
        e = b // (per_e // BLOCK)
        j0 = (b % (per_e // BLOCK)) * BLOCK
        eid[b] = e
        st[b * BLOCK:(b + 1) * BLOCK] = torch.arange(
            e * per_e + j0, e * per_e + j0 + BLOCK, dtype=torch.int32)
    npp = torch.full((1,), blocks * BLOCK, dtype=torch.int32, device=dev)
    tw = torch.rand(m * TOPK, dtype=torch.float32, device=dev)
    return codes, scales, gscale, inp, out, st, eid, npp, tw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    import vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401
    from vllm.sm70_nvfp4_ops import qpn2_moe_gemm, qpn2_prepare_moe

    dev = "cuda"
    print(f"# gpu={torch.cuda.get_device_name(0)} chunk={args.chunk} "
          f"top_k={TOPK} experts={E} block={BLOCK} "
          f"WIDE={os.environ.get('VLLM_SM70_MOE_QPN2_WIDE','?')} "
          f"WARPS={os.environ.get('VLLM_SM70_MOE_QPN2_WARPS','?')} "
          f"NS={os.environ.get('VLLM_SM70_MOE_QPN2_NS','?')} "
          f"KSPLIT={os.environ.get('VLLM_SM70_MOE_QPN2_KSPLIT','?')}")
    print(f"{'proj':>5} {'n':>6} {'k':>6} {'TFLOP':>7} {'ms':>7} "
          f"{'TFLOPS':>7} {'%peak':>6}  finite")
    tot_us = 0.0
    for name, n, k, shards in (("w13", 1024, 2048, 2), ("w2", 2048, 512, 1)):
        codes_raw, scales_raw, gscale, inp, out, st, eid, npp, tw = build(
            args.chunk, n, k, shards, dev)
        codes, scales = qpn2_prepare_moe(codes_raw, scales_raw)

        def call():
            qpn2_moe_gemm(out, inp, codes, scales, gscale, st, eid, npp, tw,
                          TOPK, mul_topk_weights=True, num_shards=shards,
                          moe_block_size=BLOCK, fuse_act=False)

        call()
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out.float()).all()) and bool(out.abs().sum() > 0)
        t = graph_time(call)
        fl = 2 * args.chunk * TOPK * n * k
        tot_us += t
        print(f"{name:>5} {n:>6} {k:>6} {fl/1e12:>7.3f} {t/1e3:>7.3f} "
              f"{fl/(t*1e-6)/1e12:>7.1f} {fl/(t*1e-6)/1e12/PEAK_TFLOPS*100:>5.1f}% "
              f" {str(ok):>5}")
    per_layer_ms = tot_us / 1e3
    print(f"\nper MoE layer (w13+w2): {per_layer_ms:.3f} ms; "
          f"x40 layers x {250000 // args.chunk} chunks = "
          f"{per_layer_ms * 40 * (250000 // args.chunk) / 1e3:.2f} s "
          f"(profile measured 17.03 s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
