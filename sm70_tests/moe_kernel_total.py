#!/usr/bin/env python3
"""Extract per-kernel GPU time for the MoE expert GEMM from a prefill trace.

The serving-path ruler (250K ttft) has a 1.4% spread, which cannot resolve a
change that moves ~1.1 s of kernel work out of 143 s.  The kernel itself is
measured over ~2800 launches with real router statistics, so its total is both
higher-resolution and more relevant than end-to-end wall time.

Prints the MoE wide/parent kernel totals plus the window's GPU-busy time, so
two launcher arms can be compared on identical real workload.

    python sm70_tests/moe_kernel_total.py <dir-or-trace> [<dir-or-trace> ...]
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

TARGETS = ("nvfp4_qpn2_moe_wide_kernel", "nvfp4_qpn2_moe_sm70_kernel",
           "gdn_forward_vlk_varlen", "fused_softmax_reg")


def traces(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "dp0*.trace.json.gz")))
    return [path]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        for t in traces(arg):
            with gzip.open(t, "rt") as fh:
                ev = [e for e in json.load(fh)["traceEvents"]
                      if e.get("cat") == "kernel"]
            busy = sum(float(e.get("dur", 0) or 0) for e in ev) / 1e6
            span = (max(e["ts"] + e.get("dur", 0) for e in ev)
                    - min(e["ts"] for e in ev)) / 1e6
            print(f"--- {os.path.basename(t)}")
            print(f"    window {span:.1f} s wall, {busy:.1f} s GPU-busy, "
                  f"{len(ev)} kernel events")
            for tgt in TARGETS:
                by_grid: dict[tuple, list[float]] = {}
                for e in ev:
                    if tgt in e["name"]:
                        by_grid.setdefault(
                            tuple(e.get("args", {}).get("grid", [])),
                            []).append(float(e.get("dur", 0) or 0))
                if not by_grid:
                    continue
                tot_all = sum(sum(v) for v in by_grid.values()) / 1e6
                print(f"    {tgt:30s} {tot_all:8.2f} s total")
                for grid, sel in sorted(by_grid.items(),
                                        key=lambda kv: -sum(kv[1])):
                    print(f"        grid={str(grid):>16} n={len(sel):6d} "
                          f"sum={sum(sel)/1e6:7.2f} s "
                          f"mean={sum(sel)/len(sel)/1e3:8.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
