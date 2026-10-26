#!/usr/bin/env bash
# W6: in-service decode profile (65K prefix, conc1) under wire=3/W3 kernel.
set -uo pipefail
PDIR=/root/run/prof
MARK=$PDIR/.w6_marker
rm -f "$MARK"
echo "=== W6 DECODE PROFILE $(date +%T) ==="
curl -s --max-time 10 -X POST http://127.0.0.1:8001/start_profile; echo " <- start_profile $(date +%T)"
touch "$MARK"
sleep 3
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
PY=/data/vllm-dev/bin/python
$PY qsa_conc_tput.py --mode decode --conc 1 --len 65536 --salt-seed 9902 --rounds 2 2>&1 | tail -3
curl -s --max-time 10 -X POST http://127.0.0.1:8001/stop_profile; echo " <- stop_profile $(date +%T)"
for i in $(seq 1 36); do
  n=$(find "$PDIR" -maxdepth 1 -name "*rank0*.pt.trace.json.gz" -newer "$MARK" 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] && break
  sleep 5
done
echo "new rank0 traces: $n"
ls -lat --time-style=+%H:%M:%S "$PDIR"/*rank0*.pt.trace.json.gz 2>/dev/null | head -2 | awk '{print "  "$6" "$5" "$7}'
echo "=== W6 DONE $(date +%T) ==="
