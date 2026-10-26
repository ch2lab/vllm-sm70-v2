#!/bin/bash
# Recovery 2026-10-14: bring PLE daemon up AS ITS NEW SERVICE, then serve.
# Sequential health gates: 8002 (daemon) -> 8001 (serve). No pattern kills.
set -x
OUT=/data/sm70_results/round5/recovery
mkdir -p $OUT

echo "=== daemon start $(date) ==="
systemctl enable --now sm70-ple-daemon
DOK=0
for i in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8002/health 2>/dev/null)
  if [ "$code" = "200" ]; then DOK=1; echo "DAEMON READY ~$((i*15))s"; break; fi
  systemctl is-active -q sm70-ple-daemon || { echo "FATAL: daemon unit inactive at loop $i"; systemctl status sm70-ple-daemon --no-pager | head -15; break; }
  sleep 15
done
[ $DOK -eq 1 ] || { echo "FATAL: daemon not ready in 25min"; exit 1; }

echo "=== serve start $(date) ==="
systemctl start sm70-serve
SOK=0
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/health 2>/dev/null)
  if [ "$code" = "200" ]; then SOK=1; echo "SERVE HEALTH_OK ~$((i*10))s"; break; fi
  systemctl is-active -q sm70-serve || { echo "NOTE: serve unit inactive at loop $i, restarting once"; systemctl reset-failed sm70-serve; systemctl start sm70-serve; }
  sleep 10
done
[ $SOK -eq 1 ] || echo "FATAL: serve not healthy in 20min"

echo "=== final state ==="
systemctl is-active sm70-ple-daemon sm70-serve
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== recovery end $(date) ==="
