#!/bin/bash
# sm70-daemon readiness gate (used as ExecStartPost): the daemon unit leaves
# "activating" only when the weight-cache endpoint (8002) is healthy
# (preload complete). Bounded: 30 min -> exit 1 (unit start fails,
# Restart=always retries; StartLimit bounds the loop).
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8002/health 2>/dev/null)
  [ "$code" = "200" ] && exit 0
  sleep 15
done
echo "sm70-daemon readiness gate: 8002 not healthy in 30min" >&2
exit 1
