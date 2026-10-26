#!/bin/bash
# AR window #3: L1 codec arms on the winning route (hier, mode=1).
# Window #2 settled: L2 hier route wins (decode 65.0 vs 55.0 tps, 6/6,
# prefill flat) and is now IN PRODUCTION (mode 1 + wire 0). This window
# tests the wire codecs on that route, per-layer (ar-three-layer-spec L1):
#   arm e4m3   = VLLM_SM70_AR_WIRE=1 (per-128 e4m3)  — window #1 broke
#                content on this path (4/6, empty completions): retest on
#                the clean per-arm-restart protocol
#   arm nvfp4  = VLLM_SM70_AR_WIRE=2 (e2m1 + per-16 e4m3) — new port
# Judge: battery 6/6 (correctness gate first) + decode tps (winner vs
# wire=0 baseline 65.0). Per-arm RESTART (wire is boot-time) clears the
# prefix cache. End state: WIRE=0 (verified production config), mode=1.
# Daemon: read-only baseline + liveness check (untouchable).
set -uo pipefail
OUT=/data/sm70_results/round5/ar_hier_ab/w3
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
  # fail-condition: service failed -> surface immediately
  systemctl is-active sm70-serve >> boot_health.txt 2>&1 || true
}
trap cleanup EXIT

MODEFILE=/tmp/ar_hier_mode

boot_arm() {  # $1 = wire value, $2 = label
  echo $2 > boot_health.txt
  cat > /etc/systemd/system/sm70-serve.service.d/sm70_ar.conf <<EOF
[Service]
Environment=VLLM_SM70_AR_HIER=1
Environment=VLLM_SM70_AR_HIER_FILE=/tmp/ar_hier_mode
Environment=VLLM_SM70_AR_WIRE=$1
EOF
  systemctl daemon-reload
  echo 1 > $MODEFILE
  systemctl restart sm70-serve
  local tries=0
  while [ $tries -lt 45 ]; do
    if systemctl is-active -q sm70-serve; then
      if curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1; then
        break
      fi
    else
      # boot failed (transient SEGV seen 02:22) -> one auto-retry
      echo "boot failed (is-active), retry $tries"; tries=$((tries+1))
      sleep 20
      systemctl restart sm70-serve
    fi
    tries=$((tries+1))
    sleep 10
  done
  sleep 20
  echo "wire journal: $(journalctl -u sm70-serve --since '2 min ago' --no-pager 2>/dev/null | grep -c 'route ACTIVE') ACTIVE lines"
  journalctl -u sm70-serve --since "2 min ago" --no-pager 2>/dev/null \
    | grep -E "route ACTIVE|FAILING CLOSED" | head -2 > journal_$2.txt
  export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
}

measure() {  # $1 = label
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

echo "=== ARM e4m3 (WIRE=1) $(date) ==="
boot_arm 1 e4m3
measure e4m3

echo "=== ARM nvfp4 (WIRE=2) $(date) ==="
boot_arm 2 nvfp4
measure nvfp4

# end state: WIRE=0 (production-verified), mode=1 (hier route, in prod)
boot_arm 0 wire0_final

DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: daemon pids dead:$DAEMON_DEAD" \
  || echo "daemon liveness OK (untouched)"
echo "=== window #3 end $(date) ==="
exit 0
