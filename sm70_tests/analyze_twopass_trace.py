"""BN-aware trace analyzer for the two-pass prefill (replaces the hardcoded
(32,256,1) grid check in bn_sweep.sh, which is wrong for BN != 4096: QK^T is
[M,256]x[256,BN] so grid.x = BN/128, and BN=2048 was falsely REFUSED).

Gate instead: a real 250K two-pass prefill must show GPU-busy >= 100 s AND the
three signature kernels (bridge_kv / fused_softmax_reg / fused_update), which
only exist in that path.  Reports per-family sums, launches, and the observed
s884gemm grids so the shapes can be eyeballed.
"""
import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
bn_expected = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
# TP1-calibrated 100 s would falsely REFUSE a real TP2 (~73 s) or TP4 (~37 s)
# prefill, since the point of TP is to cut that number.
min_busy = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0

ev = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"]
      if e.get("cat") == "kernel"]
busy = sum(float(e.get("dur", 0) or 0) for e in ev) / 1e6
fam = collections.defaultdict(float)
vard = collections.defaultdict(float)
cnt = collections.Counter()
gemm_grids = collections.Counter()
sm_variants = collections.Counter()
br_variants = collections.Counter()
br_vard = collections.defaultdict(float)
other = collections.defaultdict(float)
othern = collections.Counter()
for e in ev:
    n = e["name"]
    a = e.get("args", {}) or {}
    g = tuple(a.get("grid", []) or [])
    d = float(e.get("dur", 0) or 0)
    if "fused_softmax_reg" in n:
        fam["softmax"] += d
        cnt["softmax"] += 1
        # effect proof: the template args name the variant that actually ran
        m = re.search(r"fused_softmax_reg<(\d+),\s*(\d+)>", n)
        key = f"<{m.group(1)},{m.group(2)}>" if m else "?"
        sm_variants[key] += 1
        vard[key] += d
    elif "fused_update" in n:
        fam["update"] += d
        cnt["update"] += 1
    elif "bridge_kv" in n:
        fam["bridge_kv"] += d
        cnt["bridge_kv"] += 1
        # effect proof: the template argument names the dequant that actually ran.
        # A bool template parameter demangles as true/false, not 0/1.
        m = re.search(r"bridge_kv<(\w+)>", n)
        key = f"bridge_kv<{m.group(1)}>" if m else "bridge_kv<?>"
        br_variants[key] += 1
        br_vard[key] += d
    elif "s884gemm" in n or "cutlass" in n or "gemm" in n.lower():
        fam["attn_gemm"] += d
        cnt["attn_gemm"] += 1
        if d > 200:                       # only the big attention GEMMs
            gemm_grids[(g, n.split("(")[0][:34])] += 1
    elif "moe_wide_kernel" in n:
        fam["moe_wide"] += d
        cnt["moe_wide"] += 1
    elif "nccl" in n.lower() or "allreduce" in n.lower():
        # the cost TP introduces; invisible at TP1 and the first thing to watch
        # when judging whether a larger TP is actually paying for itself
        fam["nccl"] += d
        cnt["nccl"] += 1
    else:
        # Everything the named buckets do not claim. Without this the family
        # table silently under-counts: at TP4 the named buckets summed to 33.6 s
        # of a 41.0 s busy time, and comparing that against TP1's 136.8 s of 140.4
        # produced the impossible conclusion that non-attention work *grew* with
        # TP. Anything left here is either an unmodelled stage or a name pattern
        # that stopped matching, so it must be printed, not dropped.
        other[n.split("(")[0][:58]] += d
        othern[n.split("(")[0][:58]] += 1

sig = {"bridge_kv", "softmax", "update"} <= set(cnt)
ok = busy >= min_busy and sig
print(f"  GPU-busy {busy:6.1f} s / {len(ev)} launches   "
      f"{'OK' if ok else f'*** REFUSED (need >= {min_busy:.0f} s busy and the 3 signature kernels) ***'}")
if ok:
    for k in ("attn_gemm", "softmax", "update", "bridge_kv", "moe_wide", "nccl"):
        # dur is microseconds, so per-call is ms here -- it was labelled us,
        # which made a 1.35 ms softmax read as "1.4 us"
        avg = fam[k] / cnt[k] / 1e3 if cnt[k] else 0.0
        print(f"    {k:11} {fam[k]/1e6:8.2f} s  n={cnt[k]:7}  {avg:8.3f} ms/call")
    named = sum(fam.values())
    print(f"    {'NAMED':11} {named/1e6:8.2f} s  ({100*named/(busy*1e6):5.1f}% of busy)")
    for nm, dd in sorted(other.items(), key=lambda kv: -kv[1])[:12]:
        print(f"      other {dd/1e6:7.2f} s  n={othern[nm]:7}  "
              f"{dd/othern[nm]/1e3:8.3f} ms/call  {nm}")
    print(f"    {'UNNAMED':11} {sum(other.values())/1e6:8.2f} s  "
          f"({100*sum(other.values())/(busy*1e6):5.1f}% of busy) across "
          f"{len(other)} distinct kernels")
    print(f"    expected QK^T grid.x = BN/128 = {bn_expected // 128}")
    for k, c in sm_variants.most_common():
        print(f"    softmax variant {k:10} n={c:7}  {vard[k]/1e6:7.2f} s")
    for k, c in br_variants.most_common():
        print(f"    {k:19} n={c:7}  {br_vard[k]/1e6:7.2f} s"
              f"  {br_vard[k]/c/1e3:7.3f} ms/call")
    for (g, nm), c in gemm_grids.most_common(6):
        print(f"      grid{g} x{c:<6} {nm}")
