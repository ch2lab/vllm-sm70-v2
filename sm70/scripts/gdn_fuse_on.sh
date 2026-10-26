#!/usr/bin/env bash
# Single-variable experiment: FUSED_GDN_INPUT 0 -> 1 (standard mode).
# History 2026-10-21: +1.4% standard decode with this valve (packed qkvz+ba,
# 36 GDN layers), gated (canary 9/9, long-content, quality 6/6, Xid 0).
# Activation canary: journal line "Prepared 36 ... fused ... GDN inputs".
set -uo pipefail
source /data/src/sm70_env.sh
cd /root/run

BAK="run_sm70_serve_sync.sh.bak_nofuse_gdn_$(date +%Y%m%d_%H%M)"
cp run_sm70_serve_sync.sh "$BAK"
/data/vllm-dev/bin/python - <<'PY'
import pathlib
p = pathlib.Path('/root/run/run_sm70_serve_sync.sh')
s = p.read_text()
old = "export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=0"
new = "export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1"
assert s.count(old) == 1, f"valve line count = {s.count(old)}"
p.write_text(s.replace(old, new))
print("valve flipped to 1")
PY
/root/run/ram_guard.sh 30 gdn_fuse_on
systemctl restart sm70-serve
timeout 1500 bash /root/run/serve_wait.sh 1400 sm70-serve || exit 1
echo "=== activation canary ==="
journalctl -u sm70-serve --since "$(date -d '8 min ago' '+%Y-%m-%d %H:%M:%S')" --no-pager \
  | grep -a "fused checkpoint-FP16 GDN inputs\|GDN input opt-in found no targets" | head -3
echo "=== re-anchor decode1 (salt 9503, 6 rounds) ==="
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
OUT=/data/sm70_results/stdmap
mkdir -p "$OUT"
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds 6 > "$OUT/decode1_fused.log" 2>&1
grep -E "^round=" "$OUT/decode1_fused.log" | grep -v warm | sed 's/^/  /'
$PY - "$OUT/decode1_fused.log" <<'PY'
import re, sys, statistics
vals = []
for line in open(sys.argv[1]).read().splitlines():
    if "mode=decode" in line and "warm" not in line:
        m = re.search(r"decode_agg=([0-9.]+)", line)
        if m: vals.append(float(m.group(1)))
if vals:
    print(f"  decode1(fused): n={len(vals)} median={statistics.median(vals):.1f} min={min(vals):.1f} vals={[f'{v:.1f}' for v in vals]}")
PY
