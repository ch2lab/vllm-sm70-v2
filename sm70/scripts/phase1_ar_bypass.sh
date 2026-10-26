#!/usr/bin/env bash
# AR marginal-cost A/B in no-MTP: full bypass (local partial, numerically wrong
# but structurally identical) vs the validated baseline (68.3 @ salt 9503).
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/phase1-ar-bypass
mkdir -p "$OUT"
cd /root/run
cp -a run_sm70_serve_sync_nomtp.sh "$OUT/script.bak"
/data/vllm-dev/bin/python - <<'PY'
import pathlib
p = pathlib.Path('/root/run/run_sm70_serve_sync_nomtp.sh')
s = p.read_text()
assert 'VLLM_SM70_AR_BYPASS' not in s
old = "export VLLM_SM70_AR_WIRE=3"
new = "export VLLM_SM70_AR_WIRE=3\nexport VLLM_SM70_AR_BYPASS=1"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("AR bypass armed")
PY
systemctl stop sm70-serve && sleep 2
/root/run/ram_guard.sh 30 arbypass || exit 3
systemctl start sm70-serve
timeout 1500 bash /root/run/serve_wait.sh 1400 sm70-serve || exit 1
echo "=== decode1 (salt 9503, 5 rounds) — AR bypassed ==="
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds 6 > "$OUT/decode1_9503_bypass.log" 2>&1
grep -E "^round=" "$OUT/decode1_9503_bypass.log" | grep -v warm | sed 's/^/  /'
$PY - "$OUT/decode1_9503_bypass.log" <<'PY'
import re, sys, statistics
vals = [float(m.group(1)) for line in open(sys.argv[1])
        if (m := re.search(r"decode_agg=([0-9.]+)", line)) and "warm" not in line]
print(f"  decode1 BYPASS: n={len(vals)} median={statistics.median(vals):.1f} "
      f"min={min(vals):.1f} vals={[f'{v:.1f}' for v in vals]}")
PY
