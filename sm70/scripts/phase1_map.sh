#!/usr/bin/env bash
# Fresh no-MTP kernel map: 10 s torch-profiler window fully inside steady decode.
# Fix vs stdmap_decode.sh: JSON content-type on both profile endpoints, stop
# timeout long enough for 8-rank trace flush, window at t+8..t+18 of a 2048-token
# decode round (prefill is APC-hit ~0.7 s; decode lasts ~30 s at ~66 tok/s).
set -uo pipefail
source /data/src/sm70_env.sh
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
OUT=/data/sm70_results/phase1-map
mkdir -p "$OUT"

echo "=== single 2048-token decode round, profile window t+8..t+18 ==="
$PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9507 --rounds 1 --max-tokens 2048 > "$OUT/map_run.log" 2>&1 &
RPID=$!
sleep 8
curl -sS -o /dev/null -w 'start_profile -> HTTP %{http_code}\n' --max-time 30 \
  -X POST -H "Authorization: Bearer $SM70_API_KEY" -H "Content-Type: application/json" \
  -d '{}' http://127.0.0.1:8001/start_profile
sleep 10
curl -sS -o /dev/null -w 'stop_profile -> HTTP %{http_code}\n' --max-time 120 \
  -X POST -H "Authorization: Bearer $SM70_API_KEY" -H "Content-Type: application/json" \
  -d '{}' http://127.0.0.1:8001/stop_profile
wait $RPID
tail -2 "$OUT/map_run.log" | sed 's/^/  /'

echo "=== rank0 kernel map (top 25) ==="
sleep 5
T=$(ls -t /root/run/prof/dp0_pp0_tp0_*rank0*.profiler_out.txt 2>/dev/null | head -1)
echo "  trace: $T ($(stat -c %s "$T" 2>/dev/null) bytes)"
$PY /root/run/prof_kernel.py "$T" 25 2>/dev/null | sed 's/^/  /'
