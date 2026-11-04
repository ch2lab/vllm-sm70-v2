#!/usr/bin/env python3
"""Read a vLLM torch-profiler trace and report per-decode-step GPU kernel time.

Two things make a naive aggregation wrong, and both were hit on 2026-11-04:
  * ``dur`` is microseconds, not milliseconds;
  * the window contains a prefill as well as the decode steps, and the prefill
    kernels (two-pass softmax, GDN varlen forward, the wide MoE kernel) are
    called a couple of times per layer, not once per step, so dividing their
    cost by the step count silently attributes prefill work to decode.

The step count is recovered from kernels that provably run once per layer per
step (the MoE router top-k runs on all 40 layers), and a kernel is only counted
as decode work when its call count is an integer multiple of that.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os

ROUTER = "_sm70_topk_softmax_kernel"


def load(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        d = json.load(fh)
    evs = d["traceEvents"] if isinstance(d, dict) else d
    return [e for e in evs if e.get("cat") == "kernel"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--top", type=int, default=28)
    args = ap.parse_args()

    ker = load(args.trace)
    tot: dict[str, float] = collections.defaultdict(float)
    cnt: dict[str, int] = collections.Counter()
    for e in ker:
        tot[e["name"]] += float(e.get("dur", 0) or 0)
        cnt[e["name"]] += 1
    if not tot:
        print("no kernel events")
        return 1

    rname = next((k for k in tot if ROUTER in k), None)
    if rname is None:
        print(f"cannot infer step count: {ROUTER} absent")
        return 1
    # The router runs on every layer of every step, but a step can straddle the
    # window edge, so search a small range and keep the value that makes the most
    # kernel counts land on exact multiples.
    def ragged(steps: int) -> float:
        return sum(abs(n - round(n / steps) * steps) / max(1, n)
                   for n in cnt.values() if n > 2 * steps)
    steps = min(range(max(1, round(cnt[rname] / args.layers) - 3),
                      round(cnt[rname] / args.layers) + 4), key=ragged)
    print(f"file            {os.path.basename(args.trace)}")
    print(f"kernel events   {len(ker)}   inferred decode steps {steps} "
          f"({cnt[rname]} router calls / {args.layers} layers, ragged={ragged(steps):.4f})")
    span = (max(e['ts'] + e.get('dur', 0) for e in ker)
            - min(e['ts'] for e in ker)) / 1e6
    busy = sum(tot.values()) / 1e6
    print(f"window          {span:.2f} s wall, {busy:.2f} s GPU-busy "
          f"({busy/span*100:.0f}% busy)\n")

    rows = []
    prefill = 0.0
    dec_tot = 0.0
    for name, dur in tot.items():
        n = cnt[name]
        per = round(n / steps)
        # a decode kernel is launched a whole number of times per step; anything
        # called only a handful of times in total belongs to the prefill
        if per >= 1 and abs(n - per * steps) <= max(2, 0.01 * n):
            dec_tot += dur
            rows.append((dur / 1e3 / steps, dur, per, name))
        else:
            prefill += dur
    rows.sort(reverse=True)
    dec = sum(r[0] for r in rows)
    print(f"{'ms/step':>8} {'%dec':>6} {'n/stp':>6}  kernel")
    for ms, dur_us, per, name in rows[:args.top]:
        print(f"{ms:>8.3f} {dur_us/dec_tot*100:>5.1f}% {per:>6}  {name[:88]}")
    print(f"{dec:>8.3f}  100%        = decode kernel time per step")
    print(f"  (non-per-step kernels, i.e. prefill/graph-replay artifacts: "
          f"{prefill/1e6:.2f} s total)")

    # group into functional buckets so the target is unambiguous
    buckets = {
        "MoE (qpn/nvfp4 gemv)": ("qpn", "moe"),
        "attention decode": ("decode_xqa", "combine_decode", "fa_", "fmha"),
        "GDN linear-attn": ("gdn", "gated_delta", "conv1d"),
        "dense GEMM/gemv": ("gemv", "gemm", "cutlass", "splitK", "volta_", "sgemm"),
        "norm/act/quant glue": ("norm", "silu", "sigmoid", "dequant", "Float8",
                                "topk_softmax", "align_block", "elementwise"),
    }
    agg: dict[str, float] = collections.defaultdict(float)
    for ms, _, _, name in rows:
        low = name.lower()
        for b, keys in buckets.items():
            if any(k.lower() in low for k in keys):
                agg[b] += ms
                break
        else:
            agg["other"] += ms
    print("\nper-step by function:")
    for b, ms in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {ms:>7.3f} ms  {ms/dec*100:>5.1f}%  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
