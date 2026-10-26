#!/bin/bash
# AR window #5: FINAL codec selection A/B on the hier route (mode=1).
# Arms (env ARMS="1 2 3" default; pass the unit-PASS subset after rerun4):
#   wire=1 e4m3 (0.5 B/elem) | wire=2 nvfp4 (0.5 B/elem + scales) | wire=3 fp16 (2 B/elem)
# Per arm: boot -> battery 6/6 (content gate) -> ruler 65K (decode tps).
# Judge: highest decode tps among battery-6/6 arms.
# End state: WIRE=0 (production config), mode=1.
set -uo pipefail
ARMS="${ARMS:-1 2 3}"
OUT=/data/sm70_results/round5/ar_hier_ab/w5
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

boot_arm() {  # $1 = wire value, $2 = label
  cat > /etc/systemd/system/sm70-serve.service.d/sm70_ar.conf <<CFG
[Service]
Environment=VLLM_SM70_AR_HIER=1
Environment=VLLM_SM70_AR_HIER_FILE=/tmp/ar_hier_mode
Environment=VLLM_SM70_AR_WIRE=$1
CFG
  systemctl daemon-reload
  echo 1 > $MODEFILE
  systemctl restart sm70-serve
  local tries=0
  while [ $tries -lt 45 ]; do
    if systemctl is-active -q sm70-serve && \
       curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1; then
      break
    fi
    if ! systemctl is-active -q sm70-serve; then
      echo "boot failed (is-active), retry $tries"
      systemctl restart sm70-serve
    fi
    tries=$((tries+1))
    sleep 10
  done
  sleep 20
  journalctl -u sm70-serve --since "2 min ago" --no-pager 2>/dev/null \
    | grep -E "route ACTIVE|FAILING CLOSED" | head -2 > journal_$2.txt
  cat journal_$2.txt
  export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
}

LABELS=(unused e4m3 nvfp4 fp16)
for w in $ARMS; do
  L=${LABELS[$w]}
  echo "=== ARM $L (WIRE=$w) $(date) ==="
  boot_arm "$w" "$L"
  echo "=== arm $L: battery $(date) ==="
  timeout 3600 /data/vllm-dev/bin/python \
    /data/src/vllm-v2/sm70_tests/check_online_content.py > battery_$L.log 2>&1
  echo "BATTERY_${L}_RC=$?"
  tail -2 battery_$L.log
  echo "=== arm $L: ruler 65K $(date) ==="
  ( cd /data/src/vllm-v2 && timeout 900 /data/vllm-dev/bin/python \
    sm70_tests/baseline.py --lens 65536 > $OUT/ruler_$L.log 2>&1 )
  echo "RULER_${L}_RC=$?"
  grep -E "prefill=|baseline:" $OUT/ruler_$L.log
done

# end state: WIRE=0 production config, mode=1
echo "=== final boot (WIRE=0) $(date) ==="
boot_arm 0 wire0_final

DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: daemon pids dead:$DAEMON_DEAD" \
  || echo "daemon liveness OK (untouched)"
echo "=== window #5 end $(date) ==="
exit 0
