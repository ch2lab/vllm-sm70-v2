#!/usr/bin/env bash
# W4: in-service prefill profile (65K, salt 9502) -> trace-based kernel map.
set -uo pipefail
PDIR=/root/run/opt1_profiles
mkdir -p "$PDIR"
MARK=$PDIR/.w4_marker
rm -f "$MARK"
echo "=== W4 PREFILL PROFILE $(date +%T) ==="
curl -s --max-time 10 -X POST http://127.0.0.1:8001/start_profile; echo " <- start_profile $(date +%T)"
touch "$MARK"
sleep 3
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
PY=/data/vllm-dev/bin/python
echo "--- prefill 65K $(date +%T) ---"
$PY qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9702 --rounds 2 2>&1 | tail -3
curl -s --max-time 10 -X POST http://127.0.0.1:8001/stop_profile; echo " <- stop_profile $(date +%T)"
for i in $(seq 1 36); do
  n=$(find "$PDIR" -maxdepth 1 -name "*.pt.trace.json.gz" -newer "$MARK" 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && break
  sleep 5
done
echo "new traces: $n/8"
find "$PDIR" -maxdepth 1 -name "*.json.gz" -newer "$MARK" | sort | head -10
echo "=== W4 PROFILE DONE $(date +%T) ==="
