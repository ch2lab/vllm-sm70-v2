#!/usr/bin/env bash
# Phase-1 fix: the HC fusion enable predicates read `lora_rank` but the module
# exposes `hc_lowrank` -> all three HC fusions were silently dead ("found no
# targets"), leaving HC up/gate (96 calls/step) and HC down (12 calls/step) on
# the cuBLAS fallback. Restored 10-21 optimizations: +2.2% (up+gate) +0.9% (down+SiLU).
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/phase1-hc-fusion
mkdir -p "$OUT"
cd /root/run
cp -a run_sm70_serve_sync_nomtp.sh "$OUT/script.bak"
/data/vllm-dev/bin/python - <<'PY'
import pathlib
p = pathlib.Path('/root/run/run_sm70_serve_sync_nomtp.sh')
s = p.read_text()
for line in ("\nexport VLLM_SM70_ROUTE_LOG=1", "\nexport VLLM_SM70_GEMV_SHAPES=1"):
    if line in s:
        s = s.replace(line, "")
p.write_text(s)
print("diagnostic envs removed")
PY
systemctl stop sm70-serve && sleep 2
/root/run/ram_guard.sh 30 hcfusion || exit 3
systemctl start sm70-serve
timeout 1500 bash /root/run/serve_wait.sh 1400 sm70-serve || exit 1
echo "=== HC fusion canaries (INFO expected, not WARNING) ==="
journalctl -u sm70-serve --since "$(date -d '8 min ago' '+%Y-%m-%d %H:%M:%S')" --no-pager \
  | grep -aE "fused FP16 HC (up\+gate|down\+SiLU)|HC .*opt-in found no targets" | cut -c1-170 | sed 's/^/  /'
echo "=== decode1 anchor (salt 9503, 6 rounds) ==="
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds 6 > "$OUT/decode1_9503.log" 2>&1
grep -E "^round=" "$OUT/decode1_9503.log" | grep -v warm | sed 's/^/  /'
$PY - "$OUT/decode1_9503.log" <<'PY'
import re, sys, statistics
vals = [float(m.group(1)) for line in open(sys.argv[1])
        if (m := re.search(r"decode_agg=([0-9.]+)", line)) and "warm" not in line]
print(f"  decode1 HC-FIX: n={len(vals)} median={statistics.median(vals):.1f} "
      f"min={min(vals):.1f} vals={[f'{v:.1f}' for v in vals]}")
PY
echo "=== content gates ==="
SM70_RESULTS_DIR="$OUT" timeout 1200 $PY correctness.py 2>&1 | tail -12 | sed 's/^/  /'
timeout 900 $PY long_content_gate.py 2>&1 | tail -4 | sed 's/^/  /'
dmesg 2>/dev/null | grep -ci Xid | sed 's/^/  dmesg Xid lines: /'
