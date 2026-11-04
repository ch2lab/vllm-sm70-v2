#!/usr/bin/env python3
"""Per-(kernel, grid, block) attribution for a prefill trace.

Why grid+block rather than kernel name: a kernel name covers every shape it is
ever launched on, so its total time hides exactly the thing that matters.  The
session's only real diagnosis came from splitting sm70_fp8_qpn_kernel by grid
and finding a bucket with 64 CTAs on an 80-SM GPU - one CTA per SM, 55% of HBM
- which the per-kernel total showed no sign of.

ns/CTA is the column to read: for a bandwidth-bound kernel it is flat across
buckets with enough CTAs to fill the machine and inflates for the starved ones.
CTAs/launch below ~3x the SM count (80 here) is the starved regime.

    python sm70_tests/moe_kernel_total.py <dir-or-trace> [<dir-or-trace> ...]
    python sm70_tests/moe_kernel_total.py --all [--min-share 0.01] <trace>
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

TARGETS = ("nvfp4_qpn2_moe_wide_kernel", "nvfp4_qpn2_moe_sm70_kernel",
           "gdn_forward_vlk_varlen", "fused_softmax_reg")
SM_COUNT = 80
# nvfp4_qpn2_moe_wide_kernel<SplitK, RowTiles, WCols, K4Gran, NSplit, Fp16B>
# nvfp4_qpn2_moe_sm70_kernel<SplitK, NAcc, RowTiles, ACT_INPUT>
# grid.x = n / WCols (wide) or n / 32 (route-tiled); grid.y = route_tiles


def traces(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "dp0*.trace.json.gz")))
    return [path]


def parse_template(name: str) -> dict:
    """Pull the integer template args out of a demangled kernel name."""
    if "<" not in name or "(" not in name:
        return {}
    head = name[: name.index("(")]
    if "<" not in head:
        return {}
    inner = head[head.index("<") + 1: head.rindex(">")]
    out = {}
    for i, tok in enumerate(inner.replace(",", " ").split()):
        if tok.isdigit():
            out[i] = int(tok)
    return out


def bucket_rows(ev: list[dict], min_share: float) -> list[tuple]:
    tot = sum(float(e.get("dur", 0) or 0) for e in ev)
    groups: dict[tuple, list[float]] = {}
    for e in ev:
        if float(e.get("dur", 0) or 0) <= 0:
            continue
        a = e.get("args", {}) or {}
        key = (e["name"], tuple(a.get("grid", []) or []),
               tuple(a.get("block", []) or []))
        groups.setdefault(key, []).append(float(e["dur"]))
    rows = []
    for (nm, grid, blk), ds in groups.items():
        share = sum(ds) / tot if tot else 0.0
        if share < min_share:
            continue
        ctas = 1
        for g in grid:
            ctas *= max(1, int(g))
        warps = max(1, int(blk[0]) // 32) if blk else 1
        rows.append((share, sum(ds) / 1e6, len(ds), sum(ds) / len(ds),
                     ctas, ctas * warps / max(1, SM_COUNT), warps, nm, grid))
    rows.sort(reverse=True)
    return rows


def main() -> int:
    argv = sys.argv[1:]
    all_mode = False
    min_share = 0.0
    if "--all" in argv:
        argv.remove("--all")
        all_mode = True
        min_share = 0.01
    if "--min-share" in argv:
        i = argv.index("--min-share")
        min_share = float(argv[i + 1])
        del argv[i:i + 2]
        all_mode = True
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        for t in traces(arg):
            with gzip.open(t, "rt") as fh:
                ev = [e for e in json.load(fh)["traceEvents"]
                      if e.get("cat") == "kernel"]
            busy = sum(float(e.get("dur", 0) or 0) for e in ev) / 1e6
            span = ((max(e["ts"] + e.get("dur", 0) for e in ev)
                     - min(e["ts"] for e in ev)) / 1e6) if ev else 0.0
            print(f"--- {os.path.basename(t)}")
            print(f"    window {span:.1f} s wall, {busy:.1f} s GPU-busy, "
                  f"{len(ev)} kernel events")
            if all_mode:
                print(f"    {'share':>6} {'total_s':>8} {'n':>7} "
                      f"{'mean_us':>8} {'CTAs':>7} {'warp/SM':>8} {'thr':>5}"
                      f"  grid / kernel")
                for (share, tot_s, n, mean_us, ctas, wsm, warps,
                     nm, grid) in bucket_rows(ev, min_share):
                    print(f"    {share*100:5.1f}% {tot_s:8.2f} {n:7d} "
                          f"{mean_us:8.1f} {ctas:7d} {wsm:8.1f} {warps:5d}"
                          f"  {str(grid):>16} {nm[:74]}")
                continue
            for tgt in TARGETS:
                by_grid: dict[tuple, list[float]] = {}
                for e in ev:
                    if tgt in e["name"]:
                        a = e.get("args", {}) or {}
                        by_grid.setdefault(
                            (tuple(a.get("grid", []) or []),
                             tuple(a.get("block", []) or [])), []).append(
                                float(e.get("dur", 0) or 0))
                if not by_grid:
                    continue
                tot_all = sum(sum(v) for v in by_grid.values()) / 1e6
                print(f"    {tgt:30s} {tot_all:8.2f} s total")
                for (grid, blk), sel in sorted(by_grid.items(),
                                               key=lambda kv: -sum(kv[1])):
                    ctas = 1
                    for g in grid:
                        ctas *= max(1, int(g))
                    print(f"        grid={str(grid):>16} block={str(blk):>14}"
                          f" n={len(sel):6d} sum={sum(sel)/1e6:7.2f} s "
                          f"mean={sum(sel)/len(sel)/1e3:8.3f} ms "
                          f"ns/CTA={sum(sel)*1e3/len(sel)/max(1,ctas):7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
