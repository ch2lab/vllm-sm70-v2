#!/usr/bin/env bash
# Iter-8 profile window 2: conc8 STEADY decode only (rounds 2 = warm + 1 timed),
# torch profiler on live serve, 8K production state.
set -uo pipefail
PDIR=/root/run/opt1_profiles
MARK=$PDIR/.iter8c8_marker
rm -f "$MARK"
echo "=== ITER8 PROFILE-CONC8 $(date +%T) ==="
curl -s --max-time 10 -X POST http://127.0.0.1:8001/start_profile; echo " <- start_profile $(date +%T)"
touch "$MARK"
sleep 3
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
PY=/data/vllm-dev/bin/python
$PY qsa_conc_tput.py --mode decode --conc 8 --len 1024 --salt-seed 961001 --rounds 2 2>&1 | tail -3
curl -s --max-time 10 -X POST http://127.0.0.1:8001/stop_profile; echo " <- stop_profile $(date +%T)"
for i in $(seq 1 36); do
  n=$(find $PDIR -maxdepth 1 -name "*.pt.trace.json.gz" -newer "$MARK" 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && break
  sleep 5
done
echo "new traces: $n/8"
find $PDIR -maxdepth 1 -name "*.json.gz" -newer "$MARK" | sort | head -10
echo "=== ITER8 PROFILE-CONC8 DONE $(date +%T) ==="
