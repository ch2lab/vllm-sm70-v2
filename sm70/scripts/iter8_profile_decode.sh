#!/usr/bin/env bash
# Iter-8 profile window: decode axis (conc1 + conc8) under torch profiler,
# 8K production state. Traces -> /root/run/opt1_profiles (flat, per serve config).
set -uo pipefail
PDIR=/root/run/opt1_profiles
MARK=$PDIR/.iter8_marker
rm -f "$MARK"
echo "=== ITER8 PROFILE $(date +%T) ==="
curl -s --max-time 10 -X POST http://127.0.0.1:8001/start_profile; echo " <- start_profile $(date +%T)"
touch "$MARK"
sleep 3
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
PY=/data/vllm-dev/bin/python
echo "--- probe decode1 $(date +%T) ---"
$PY qsa_conc_tput.py --mode decode --conc 1 --len 1024 --salt-seed 9502 --rounds 1 2>&1 | tail -2
echo "--- probe conc8 $(date +%T) ---"
$PY qsa_conc_tput.py --mode decode --conc 8 --len 1024 --salt-seed 961001 --rounds 1 2>&1 | tail -2
curl -s --max-time 10 -X POST http://127.0.0.1:8001/stop_profile; echo " <- stop_profile $(date +%T)"
# bounded flush wait: expect 8 rank traces newer than marker
for i in $(seq 1 36); do
  n=$(find $PDIR -maxdepth 1 -name "*.pt.trace.json.gz" -newer "$MARK" 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && break
  sleep 5
done
echo "new traces: $n/8"
find $PDIR -maxdepth 1 -name "*.json.gz" -newer "$MARK" | sort | head -10
find $PDIR -maxdepth 1 -name "*.profiler_out.txt" -newer "$MARK" | sort | head -10
echo "=== ITER8 PROFILE DONE $(date +%T) ==="
