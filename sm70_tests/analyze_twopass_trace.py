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
import sys

path = sys.argv[1]
bn_expected = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

ev = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"]
      if e.get("cat") == "kernel"]
busy = sum(float(e.get("dur", 0) or 0) for e in ev) / 1e6
fam = collections.defaultdict(float)
cnt = collections.Counter()
gemm_grids = collections.Counter()
for e in ev:
    n = e["name"]
    a = e.get("args", {}) or {}
    g = tuple(a.get("grid", []) or [])
    d = float(e.get("dur", 0) or 0)
    if "fused_softmax_reg" in n:
        fam["softmax"] += d
        cnt["softmax"] += 1
    elif "fused_update" in n:
        fam["update"] += d
        cnt["update"] += 1
    elif "bridge_kv" in n:
        fam["bridge_kv"] += d
        cnt["bridge_kv"] += 1
    elif "s884gemm" in n or "cutlass" in n or "gemm" in n.lower():
        fam["attn_gemm"] += d
        cnt["attn_gemm"] += 1
        if d > 200:                       # only the big attention GEMMs
            gemm_grids[(g, n.split("(")[0][:34])] += 1
    elif "moe_wide_kernel" in n:
        fam["moe_wide"] += d
        cnt["moe_wide"] += 1

sig = {"bridge_kv", "softmax", "update"} <= set(cnt)
print(f"  GPU-busy {busy:6.1f} s / {len(ev)} launches   "
      f"{'OK' if busy > 100 and sig else '*** REFUSED (not a real 250K prefill) ***'}")
if busy > 100 and sig:
    for k in ("attn_gemm", "softmax", "update", "bridge_kv", "moe_wide"):
        avg = fam[k] / cnt[k] / 1e3 if cnt[k] else 0.0
        print(f"    {k:11} {fam[k]/1e6:8.2f} s  n={cnt[k]:7}  {avg:7.1f} us/call")
    print(f"    expected QK^T grid.x = BN/128 = {bn_expected // 128}")
    for (g, nm), c in gemm_grids.most_common(6):
        print(f"      grid{g} x{c:<6} {nm}")
