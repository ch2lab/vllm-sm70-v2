#!/bin/bash
# Recovery tail 2026-10-14 (renamed unit): wait sm70-daemon 8002 -> start serve -> 8001.
set -x
echo "=== tail: wait daemon 8002 $(date) ==="
DOK=0
for i in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8002/health 2>/dev/null)
  [ "$code" = "200" ] && { DOK=1; echo "DAEMON READY ~$((i*15))s"; break; }
  st=$(systemctl show -p ActiveState --value sm70-daemon)
  [ "$st" = "failed" ] && { echo "FATAL: sm70-daemon in terminal failed state"; systemctl status sm70-daemon --no-pager | head -15; break; }
  sleep 15
done
[ $DOK -eq 1 ] || { echo "FATAL: daemon not ready in 25min"; exit 1; }
echo "=== serve re-attach $(date) ==="
# RESTART (not start): the running serve attached to the OLD stopped daemon
# (sm70-ple-daemon) — its IPC refs are dangling; a clean boot against the
# new sm70-daemon is required.
systemctl restart sm70-serve
SOK=0
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/health 2>/dev/null)
  [ "$code" = "200" ] && { SOK=1; echo "SERVE HEALTH_OK ~$((i*10))s"; break; }
  systemctl is-active -q sm70-serve || { echo "NOTE: serve inactive, retrying"; systemctl reset-failed sm70-serve 2>/dev/null; systemctl start sm70-serve; }
  sleep 10
done
[ $SOK -eq 1 ] || echo "FATAL: serve not healthy in 20min"
echo "=== final state ==="
systemctl is-active sm70-daemon sm70-serve
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== tail end $(date) ==="
