#!/usr/bin/env bash
# Arm C: monitor weight-cache daemon readiness (health port 8002).
# Args: $1 = daemon parent PID
for i in $(seq 1 100); do
  sleep 15
  if ! kill -0 "$1" 2>/dev/null; then
    echo "FAIL: daemon process died after ~$((i*15))s"
    tail -30 /root/run/ple_preload.log
    exit 1
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8002/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "DAEMON READY (~$((i*15))s)"
    exit 0
  fi
done
echo "TIMEOUT: daemon not ready after 25min"
tail -30 /root/run/ple_preload.log
exit 2
