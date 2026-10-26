#!/usr/bin/env bash
# Arm C: monitor IPC engine boot with OOM + host-RAM watch.
# Aborts on: service death, any GPU > 28GB HBM, MemAvailable < 20GB.
for i in $(seq 1 100); do
  sleep 15
  st=$(systemctl is-active sm70-serve 2>/dev/null)
  if [ "$st" != "active" ]; then
    echo "FAIL: engine service state=$st after ~$((i*15))s"
    journalctl -u sm70-serve --since "10 min ago" --no-pager 2>/dev/null | grep -iE "error|traceback|oom|CUDA out of memory|fallback" | tail -20
    exit 1
  fi
  peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  avail=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
  if [ "$peak" -gt 28672 ]; then
    echo "ABORT: GPU HBM peak ${peak}MiB > 28GB cap"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 3
  fi
  if [ "$avail" -lt 20 ]; then
    echo "ABORT: host MemAvailable ${avail}GB < 20GB floor"
    grep -E "MemTotal|MemFree|MemAvailable|Shmem" /proc/meminfo
    exit 4
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "ARM_C ENGINE READY (~$((i*15))s, HBM peak ${peak}MiB, host avail ${avail}GB)"
    exit 0
  fi
done
echo "TIMEOUT: engine not ready after 25min"
journalctl -u sm70-serve --since "5 min ago" --no-pager | tail -25
exit 5
