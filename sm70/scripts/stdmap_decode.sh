#!/usr/bin/env bash
# Standard-mode (no MTP) decode map + anchor, 65K ruler.
# 1) decode1 anchor (salt 9503, 6 rounds -> median/min)
# 2) one steady decode round with a 10s torch-profiler window started ~30s in
#    (prefill done, window prefill-free), then per-rank kernel map via prof_kernel.py
set -uo pipefail
source /data/src/sm70_env.sh
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
OUT=/data/sm70_results/stdmap
mkdir -p "$OUT"

echo "=== 1) decode1 anchor (salt 9503, 6 rounds) ==="
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds 6 > "$OUT/decode1_9503.log" 2>&1
grep -E "^round=" "$OUT/decode1_9503.log" | grep -v warm | sed 's/^/  /'
$PY - "$OUT/decode1_9503.log" <<'PY'
import re, sys, statistics
vals = []
for line in open(sys.argv[1]).read().splitlines():
    if "mode=decode" in line and "warm" not in line:
        m = re.search(r"decode_agg=([0-9.]+)", line)
        if m: vals.append(float(m.group(1)))
if vals:
    print(f"  decode1: n={len(vals)} median={statistics.median(vals):.1f} min={min(vals):.1f} vals={[f'{v:.1f}' for v in vals]}")
PY

echo "=== 2) steady decode + 10s profile window ==="
$PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9505 --rounds 1 --max-tokens 2048 > "$OUT/stdmap_run.log" 2>&1 &
RPID=$!
sleep 30   # let the 65K prefill (~9s) finish; window starts in steady decode
curl -s --max-time 10 -X POST -H "Authorization: Bearer $SM70_API_KEY" \
  -d '{}' http://127.0.0.1:8001/start_profile >/dev/null
echo "  start_profile $(date +%T)"
sleep 10
curl -s --max-time 10 -X POST -H "Authorization: Bearer $SM70_API_KEY" \
  -d '{}' http://127.0.0.1:8001/stop_profile >/dev/null
echo "  stop_profile $(date +%T)"
wait $RPID
tail -3 "$OUT/stdmap_run.log" | sed 's/^/  /'

echo "=== 3) rank0 kernel map ==="
sleep 5   # let the profiler flush
T=$(ls -t /root/run/prof/dp0_pp0_tp0_*rank0*.profiler_out.txt 2>/dev/null | head -1)
echo "  trace: $T"
$PY prof_kernel.py "$T" 18 | sed 's/^/  /'
