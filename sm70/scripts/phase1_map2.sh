#!/usr/bin/env bash
# Phase-1 no-MTP kernel map v2: decode-ONLY 10 s window.
# Round plan of qsa_conc_tput.py --rounds N = 1 warm round (cold 65K prefill ~9.3 s
# + 2048-token decode ~31 s) + (N-1) measured rounds (APC-hit prefill ~0.6 s + 31 s
# decode). So the measured round starts at ~t+41 s; profile inside it: t+50..t+60.
set -uo pipefail
source /data/src/sm70_env.sh
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
OUT=/data/sm70_results/phase1-map
mkdir -p "$OUT"

echo "=== rounds=2 (warm + 1 measured), 2048 tokens, profile window t+50..t+60 ==="
$PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9508 --rounds 2 --max-tokens 2048 > "$OUT/map_run_v2.log" 2>&1 &
RPID=$!
sleep 50
curl -sS -o /dev/null -w 'start_profile -> HTTP %{http_code}\n' --max-time 30 \
  -X POST -H "Authorization: Bearer $SM70_API_KEY" -H "Content-Type: application/json" \
  -d '{}' http://127.0.0.1:8001/start_profile
sleep 10
curl -sS -o /dev/null -w 'stop_profile -> HTTP %{http_code}\n' --max-time 120 \
  -X POST -H "Authorization: Bearer $SM70_API_KEY" -H "Content-Type: application/json" \
  -d '{}' http://127.0.0.1:8001/stop_profile
wait $RPID
tail -3 "$OUT/map_run_v2.log" | sed 's/^/  /'

echo "=== rank0 decode kernel map (top 30) ==="
sleep 5
T=$(ls -t /root/run/prof/dp0_pp0_tp0_*rank0*.pt.trace.json.gz | head -1)
echo "  trace: $T ($(stat -c %s "$T") bytes)"
$PY prof_kernel_gz.py "$T" 30 | tee "$OUT/kernel_map_v2.txt" | sed 's/^/  /'
