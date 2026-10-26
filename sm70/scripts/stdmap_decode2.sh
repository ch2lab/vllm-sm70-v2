#!/usr/bin/env bash
# Standard-mode (no MTP) decode kernel map — v2 (fixes v1's 400 + window bugs).
#  v1 bugs: (a) curl -d '{}' defaulted to form-urlencoded -> /start_profile 400;
#  (b) t+30..t+40 window straddled the round end (~t+31s) -> idle dilution.
# Window is now t+5..t+15 (fully inside the 2048-step decode phase, prefill
# ends t+0.6s). Round is the (single) warm round of --rounds 1.
set -uo pipefail
source /data/src/sm70_env.sh
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
OUT=/data/sm70_results/stdmap
mkdir -p "$OUT"
MARKER=$(date '+%Y-%m-%d %H:%M:%S')

$PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9505 --rounds 1 --max-tokens 2048 > "$OUT/stdmap_run2.log" 2>&1 &
RPID=$!
sleep 5
R1=$(curl -s --max-time 10 -w "%{http_code}" -X POST \
  -H "Authorization: Bearer $SM70_API_KEY" -H "Content-Type: application/json" \
  -d '{}' http://127.0.0.1:8001/start_profile)
echo "start_profile -> HTTP $R1"
sleep 10
R2=$(curl -s --max-time 10 -w "%{http_code}" -X POST \
  -H "Authorization: Bearer $SM70_API_KEY" -H "Content-Type: application/json" \
  -d '{}' http://127.0.0.1:8001/stop_profile)
echo "stop_profile -> HTTP $R2"
[ "$R1" = "200" ] && [ "$R2" = "200" ] || { echo "PROFILE API FAILED"; wait $RPID; exit 1; }
wait $RPID
tail -2 "$OUT/stdmap_run2.log" | sed 's/^/  /'

sleep 60   # let the torch profiler flush the 8x ~80MB trace dumps
T=$(find /root/run/prof -name "dp0_pp0_tp0_*rank0*.profiler_out.txt" -newermt "$MARKER" | head -1)
if [ -z "$T" ]; then echo "NO NEW rank0 profiler_out.txt"; exit 1; fi
echo "  trace: $T"
$PY prof_kernel.py "$T" 25 | sed 's/^/  /'
