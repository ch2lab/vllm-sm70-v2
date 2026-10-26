#!/bin/bash
# AR hier-route A/B window #2: ISOLATION boot (VLLM_SM70_AR_FP8_WIRE=0 = fp32 legs).
# Window #1 was INVALID: prefill arms contaminated by the prefix cache (ruler
# self-flagged invalid), and mode-1 (hier + fp8 wire) FAILED the battery 4/6
# with EMPTY completions on '4'/'mars' (content break, deterministic).
# Root cause NOT separated (route vs codec). This window isolates it:
#   arm 0 = mode 0 (NCCL path)          arm 1 = mode 1 (hier route, fp32 wire)
# Per-arm RESTART clears the prefix cache (the window #1 bug).
# Final state: mode file = 0 (NCCL, today's behavior), service left running.
# Daemon: read-only baseline + liveness check (untouchable).
set -uo pipefail
OUT=/data/sm70_results/round5/ar_hier_ab/w2
mkdir -p $OUT
cd $OUT

DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
echo "daemon pids (read-only baseline): ${DAEMON_PIDS:-NONE}"
[ -n "$DAEMON_PIDS" ] || { echo "FATAL: no daemon - refuse"; exit 1; }

cleanup() {
  if ! systemctl is-active -q sm70-serve; then
    echo "cleanup: service not active -> restarting"
    systemctl start sm70-serve
  fi
}
trap cleanup EXIT

MODEFILE=/tmp/ar_hier_mode

echo "=== install wire=0 drop-in + daemon-reload $(date) ==="
mkdir -p /etc/systemd/system/sm70-serve.service.d
cat > /etc/systemd/system/sm70-serve.service.d/sm70_ar.conf <<'EOF'
[Service]
Environment=VLLM_SM70_AR_HIER=0
Environment=VLLM_SM70_AR_HIER_FILE=/tmp/ar_hier_mode
Environment=VLLM_SM70_AR_FP8_WIRE=0
EOF
systemctl daemon-reload
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)

boot_arm() {  # $1 = mode 0|1
  echo $1 > $MODEFILE
  systemctl restart sm70-serve
  for i in $(seq 1 180); do
    curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && break
    sleep 10
  done
  sleep 20
  journalctl -u sm70-serve --since "2 min ago" --no-pager 2>/dev/null \
    | grep -E "route ACTIVE|FAILING CLOSED" | head -2 > journal_$1.txt
  echo "ACTIVE_lines=$(journalctl -u sm70-serve --since '2 min ago' --no-pager 2>/dev/null | grep -c 'route ACTIVE')"
}

measure() {  # $1 = arm label
  echo "=== arm $1: battery $(date) ==="
  ( cd $OUT && timeout 3600 /data/vllm-dev/bin/python \
      /data/src/vllm-v2/sm70_tests/check_online_content.py \
      > battery_$1.log 2>&1 )
  echo "BATTERY_$1_RC=$?"
  tail -2 battery_$1.log
  echo "=== arm $1: ruler 65K $(date) ==="
  ( cd /data/src/vllm-v2 && timeout 900 /data/vllm-dev/bin/python \
      sm70_tests/baseline.py --lens 65536 > $OUT/ruler_$1.log 2>&1 )
  echo "RULER_$1_RC=$?"
  grep -E "prefill=|baseline:" $OUT/ruler_$1.log
}

echo "=== ARM 0 (NCCL path, mode=0) $(date) ==="
boot_arm 0
measure 0

echo "=== ARM 1 (hier route, fp32 wire, mode=1) $(date) ==="
boot_arm 1
measure 1

# end state: NCCL (mode 0) on the running service - no restart needed
echo 0 > $MODEFILE
DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: daemon pids dead:$DAEMON_DEAD" \
  || echo "daemon liveness OK (untouched)"
echo "=== window #2 end $(date) ==="
exit 0
