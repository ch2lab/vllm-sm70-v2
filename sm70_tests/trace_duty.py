"""Decode-step duty cycle and gap histogram from a torch trace.

nvidia-smi says the 10K decode run holds sm ~52% at full 1530 MHz and only
137 W of 300 W, i.e. decode is NOT power- or bandwidth-limited.  The kernel-sum
attribution (~6.15 ms of kernels in a 9.74 ms step) agrees that roughly a third
of every step is not executing kernels.  This measures it directly: sort all
kernel intervals, merge overlaps, and report the coverage and the gap size
distribution, so we can tell "many small launch gaps" from "one big stall".
"""
import gzip
import json
import sys
from collections import Counter

path = sys.argv[1]
ev = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"]
      if e.get("cat") == "kernel" and e.get("dur")]

iv = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in ev)
if not iv:
    sys.exit("no kernel events")

# merge overlapping intervals (multiple streams) into busy spans
merged = []
for s, t in iv:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], t)
    else:
        merged.append([s, t])

busy = sum(t - s for s, t in merged) / 1e6
wall = (merged[-1][1] - merged[0][0]) / 1e6
gaps = [(merged[i + 1][0] - merged[i][1]) / 1e6 for i in range(len(merged) - 1)]
idle = sum(gaps)

print(f"{path.split('/')[-1]}")
print(f"  kernels {len(ev)}  merged busy spans {len(merged)}")
print(f"  window {wall:8.3f} s   busy {busy:8.3f} s ({100*busy/wall:5.1f}%)   "
      f"idle {idle:8.3f} s ({100*idle/wall:5.1f}%)")
if gaps:
    gs = sorted(gaps)
    tot = sum(gaps)
    print(f"  gaps: n={len(gaps)}  total {tot:.3f} s  median {gs[len(gs)//2]*1e3:7.1f} us  "
          f"p90 {gs[int(len(gs)*0.9)]*1e3:7.1f} us  max {gs[-1]*1e3:8.1f} us")
    buckets = Counter()
    for g in gaps:
        if g < 1e-3: buckets["<1 ms"] += 1
        elif g < 5e-3: buckets["1-5 ms"] += 1
        elif g < 20e-3: buckets["5-20 ms"] += 1
        elif g < 100e-3: buckets["20-100 ms"] += 1
        else: buckets[">100 ms"] += 1
    for b in ("<1 ms", "1-5 ms", "5-20 ms", "20-100 ms", ">100 ms"):
        n = buckets.get(b, 0)
        if n:
            print(f"    {b:9} n={n:6}")
    # where the idle actually is: top contributors
    big = sorted(range(len(gaps)), key=lambda i: -gaps[i])[:8]
    print("  largest gaps (us) and the kernel pair bounding each:")
    names = {e["ts"]: e["name"] for e in ev}
    for i in big:
        a = max((k for k in merged if k[1] == merged[i][1]), default=None)
        print(f"    {gaps[i]*1e6:9.1f}  after span ending {merged[i][1]/1e6:.3f} s")
