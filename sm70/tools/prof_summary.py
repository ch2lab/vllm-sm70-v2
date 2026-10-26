#!/usr/bin/env python
"""Summarise a torch-profiler trace: GPU kernel shares, launch counts, idle time.

Used to answer "what is the decode step actually spending time on, and is the host
the limit" without reading a 200 MB trace by hand.

Usage: prof_summary.py <trace.json.gz> [top_n]
"""

from __future__ import annotations

import collections
import gzip
import json
import sys


def main() -> int:
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    with gzip.open(path) as fh:
        trace = json.load(fh)
    events = trace["traceEvents"]

    kernels = []          # (name, dur_us)
    launches = collections.Counter()
    spans = []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        dur = e["dur"]
        spans.append((e["ts"], e["ts"] + dur))
        if cat == "kernel":
            kernels.append((name, dur))
        elif cat in ("cpu_op", "user_annotation", "python_function"):
            launches[name] += 1

    if not kernels:
        print("no kernel events")
        return 1

    start = min(s for s, _ in spans)
    end = max(e for _, e in spans)
    wall = end - start
    busy = sum(d for _, d in kernels)

    per_kernel = collections.defaultdict(lambda: [0, 0])
    for name, dur in kernels:
        per_kernel[name][0] += dur
        per_kernel[name][1] += 1

    print(f"trace: {path}")
    print(f"  wall {wall/1000:.3f} s   kernels {len(kernels)}   gpu busy {busy/1e6:.3f} s "
          f"({100*busy/wall:.1f}% of wall)   idle {(wall-busy)/1e6:.3f} s "
          f"({100*(wall-busy)/wall:.1f}%)")
    print(f"  host-side ops recorded: {sum(launches.values())} "
          f"({sum(launches.values())/max(1,len(kernels)):.2f} per kernel launch)")

    print(f"\n  top {top_n} kernels by total GPU time:")
    for name, (tot, cnt) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print(f"   {tot/1000:9.1f} ms {100*tot/busy:5.1f}%  n={cnt:6d}  "
              f"avg={tot/cnt:6.1f} us  {name[:70]}")

    print(f"\n  top {top_n} kernels by launch count:")
    for name, (tot, cnt) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:top_n]:
        print(f"   n={cnt:6d}  avg={tot/cnt:6.1f} us  tot={tot/1000:8.1f} ms  {name[:70]}")

    print("\n  busiest host-side op names (proxy for launch overhead):")
    for name, cnt in launches.most_common(12):
        print(f"   n={cnt:6d}  {name[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
