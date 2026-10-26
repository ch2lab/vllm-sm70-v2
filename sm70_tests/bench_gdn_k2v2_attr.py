#!/usr/bin/env python3
"""K2 v2 G2-FAIL ATTRIBUTION (discipline #11931: evidence before verdict).

The G2 gate measured shadow-v2 vs incumbent = ratio 1.396 (FAIL). That is a
SOMPTOM, not a cause: before any "v2 dead" ruling we need the per-kernel
breakdown of both paths (K1_wy / K2_state / K3_output on the shadow side;
the incumbent's own kernels on the other) at the identical G2 shape
(T=8192, 8-chunk pattern). torch.profiler CUDA-activity records give
per-launch kernel times with ZERO code changes.

Outputs (-> stdout + jsonl):
  * per-kernel median/sum us per path
  * which phase carries the 4169 us regression (K2 smem-blade? K3 exp-blade?
    K1/overhead?)
  * per-phase delta table — the input to the next decision (targeted fix vs
    genuine v3). Run in a quiet GPU window (after the L1 codec window).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from bench_gdn_g2_ab import make_inputs  # noqa: E402  (identical G2 inputs)
import torch  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.sm70_gdn import (  # noqa: E402
    chunk_gated_delta_rule_sm70,
    chunk_gated_delta_rule_sm70_chunked,
)

HQ, HV = 2, 6


def profile_path(name, fn, args, iters: int = 5):
    """Per-kernel CUDA time for one path (median of `iters` profiled runs)."""
    for _ in range(3):  # warm (JIT, smem opt-in, allocator)
        fn(*args)
    torch.cuda.synchronize()
    per_kernel: dict[str, list[float]] = {}
    totals: list[float] = []
    for _ in range(iters):
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        tot = 0.0
        # kernel-level records:
        for ev in prof.events():
            if ev.device_type == torch.profiler.DeviceType.CUDA and \
                    ev.self_device_time_total > 0:
                per_kernel.setdefault(ev.name, []).append(
                    ev.self_device_time_total / 1000.0)
                tot += ev.self_device_time_total / 1000.0
        totals.append(tot)
    return {
        "path": name,
        "total_us_median": round(statistics.median(totals), 1),
        "per_kernel_us_median": {
            k: round(statistics.median(v), 1) for k, v in per_kernel.items()
        },
    }


def main() -> int:
    q, k, v, gl, beta = make_inputs(8192, "cuda:0")
    cu = None  # single segment (G2 gate condition)

    # G2-contract call (same as bench_gdn_g2_ab): no final state, no
    # in-kernel l2norm (inputs pre-normalized by make_inputs), no cu,
    # no initial state — reproduces the G2 gate conditions (shadow total
    # anchored at ~10558us by the check below).
    def cur():
        return chunk_gated_delta_rule_sm70(
            q, k, v, gl, beta, None, False, None, False)

    def sh():
        return chunk_gated_delta_rule_sm70_chunked(
            q, k, v, gl, beta, None, False, None, False)

    # confirm the G2 signature reproduces (total-level) before attribution
    from bench_gdn_g2_ab import time_fn
    cur_tot = time_fn(cur, (), runs=7)
    sh_tot = time_fn(sh, (), runs=7)
    ratio = sh_tot / cur_tot
    print(f"[attr] total: incumbent={cur_tot:.1f}us shadow={sh_tot:.1f}us "
          f"ratio={ratio:.3f} (G2 reproduced: {abs(ratio-1.396)<0.1})")

    a = profile_path("incumbent", cur, ())
    b = profile_path("shadow_v2", sh, ())
    rec = {"ratio_total": round(ratio, 4), "incumbent": a, "shadow": b}

    print("\n== per-kernel breakdown (us, median) ==")
    allk = sorted(set(a["per_kernel_us_median"]) |
                  set(b["per_kernel_us_median"]))
    print(f"{'kernel':56s} {'incumbent':>12s} {'shadow_v2':>12s} {'delta':>10s}")
    for kn in allk:
        ic = a["per_kernel_us_median"].get(kn, 0.0)
        sv = b["per_kernel_us_median"].get(kn, 0.0)
        mark = ""
        if kn in ("gdn_state_k2_kernel",):
            mark += " [K2 blade: hT pad]"
        if kn in ("gdn_output_k3_kernel",):
            mark += " [K3 blade: exp split]"
        print(f"  {kn[:56]:56s} {ic:12.1f} {sv:12.1f} {sv-ic:+10.1f}{mark}")
    print(f"  {'TOTAL':56s} {a['total_us_median']:12.1f} "
          f"{b['total_us_median']:12.1f} "
          f"{b['total_us_median']-a['total_us_median']:+10.1f}")

    out = common.RESULTS_DIR / "round5" / "p3_gdn"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"k2v2_attr_{time.strftime('%Y%m%d')}.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%F %T"), **rec},
                            sort_keys=True) + "\n")
    print(f"\n[attr] written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
