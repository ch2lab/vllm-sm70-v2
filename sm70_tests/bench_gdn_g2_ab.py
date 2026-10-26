#!/usr/bin/env python3
"""P3 G2 gate — shadow vs incumbent GDN kernel A/B, identical fp16 inputs,
offline (no service traffic). Quiet window required (quiet_window_check).

Design (spec 2026-10-14-p3-gdn-shadow-kernel-spec.md, gate G2):
  - Same input tensors (T=8192 production chunk shape, TP8 per-rank heads)
    fed to BOTH kernels; CUDA-event device time, median of 7.
  - ratio = shadow / incumbent:
      <= 0.50  -> PASS
      0.50-0.75 -> MARGINAL
      > 0.75   -> FAIL
  - Patterns: "chunked" = 8x T=8192 (production 65K form),
              "single"  = 1x T=65536 (G0 may re-arbitrate the target).
No service restart. Engine may be running idle; gate enforces quiet.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import torch  # noqa: E402
from bench_qsa_kv_dtype import quiet_window_check  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (  # noqa: E402
    chunk_gated_delta_rule_sm70,
    chunk_gated_delta_rule_sm70_chunked,
)

RESULTS = common.RESULTS_DIR / "round5" / "p3_gdn"
HQ, HV, K, V = 2, 6, 128, 128


def make_inputs(T: int, dev: str, seed: int = 11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # pre-normalize q/k (l2) so both kernels see identical inputs; call with
    # use_qk_l2norm_in_kernel=False to isolate the kernel body.
    q = torch.randn(T, HQ, K, generator=g)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = torch.randn(T, HQ, K, generator=g)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.randn(T, HV, V, generator=g) * 0.1
    gl = -0.05 * torch.ones(T, HV)
    beta = 0.5 * torch.ones(T, HV)
    return (
        q.to(dev, torch.half).unsqueeze(0),
        k.to(dev, torch.half).unsqueeze(0),
        v.to(dev, torch.half).unsqueeze(0),
        gl.to(dev, torch.float32).unsqueeze(0),
        beta.to(dev, torch.float32).unsqueeze(0),
    )


def time_fn(fn, args, runs: int = 7, dev: str = "cuda:0") -> float:
    fn(*args)  # warm
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(*args)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="chunked",
                    choices=["chunked", "single"])
    ap.add_argument("--runs", type=int, default=7)
    args = ap.parse_args()

    if args.pattern == "chunked":
        T = 8192
        n_calls = 8
    else:
        T = 65536
        n_calls = 1

    ok, _ = quiet_window_check()
    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS / f"gdn_g2_ab_{time.strftime('%Y%m%d')}.jsonl"

    def rec(e):
        e = {"ts": time.strftime("%F %T"), **e}
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(e, sort_keys=True) + "\n")

    rec({"event": "quiet_window", "pass": ok, "pattern": args.pattern})
    if not ok:
        print("G2 BLOCKED: quiet window failed")
        return 2

    q, k, v, gl, beta = make_inputs(T, "cuda:0")
    cur_us = time_fn(chunk_gated_delta_rule_sm70,
                     (q, k, v, gl, beta, None, False, None, False),
                     args.runs)
    sh_us = time_fn(chunk_gated_delta_rule_sm70_chunked,
                    (q, k, v, gl, beta, None, False, None, False),
                    args.runs)
    ratio = sh_us / cur_us
    verdict = ("PASS" if ratio <= 0.50
               else "MARGINAL" if ratio <= 0.75 else "FAIL")
    rec({"event": "g2_ab", "pattern": args.pattern, "T": T,
         "n_calls": n_calls, "incumbent_us": round(cur_us, 1),
         "shadow_us": round(sh_us, 1), "ratio": round(ratio, 4),
         "verdict": verdict})
    print(f"pattern={args.pattern} T={T}: incumbent={cur_us:.1f}us "
          f"shadow={sh_us:.1f}us ratio={ratio:.3f} -> {verdict}")
    # Window discipline (2026-10-15): the exit code must reflect the verdict
    # so the calling window can gate on rc (PASS=0, MARGINAL/FAIL=1).
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
