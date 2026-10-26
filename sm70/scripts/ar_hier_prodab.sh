#!/bin/bash
# AR hierarchical-route production A/B (throughput-final per user rule 2026-10-15).
# Same-boot protocol via VLLM_SM70_AR_HIER_FILE: mode re-read per eligible AR
# launch (0 = NCCL path, 1 = hierarchical decode/verify route + fp8 wire).
# Zero-restart flip; rollback = write 0 (behavior = today).
# Daemon read-only baseline + tail check (protocol invariant).
set -uo pipefail
OUT=/data/sm70_results/round5/ar_hier_ab
mkdir -p $OUT
cd $OUT

DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
echo "daemon pids (read-only baseline): ${DAEMON_PIDS:-NONE}"
[ -n "$DAEMON_PIDS" ] || { echo "FATAL: no daemon — refuse"; exit 1; }

cleanup() {
  if ! systemctl is-active -q sm70-serve; then
    echo "cleanup: service not active -> restarting"
    systemctl start sm70-serve
  fi
}
trap cleanup EXIT

MODEFILE=/tmp/ar_hier_mode

echo "=== install drop-in + restart $(date) ==="
mkdir -p /etc/systemd/system/sm70-serve.service.d
cat > /etc/systemd/system/sm70-serve.service.d/sm70_ar.conf <<'EOF'
[Service]
Environment=VLLM_SM70_AR_HIER=0
Environment=VLLM_SM70_AR_HIER_FILE=/tmp/ar_hier_mode
Environment=VLLM_SM70_AR_FP8_WIRE=1
EOF
echo 0 > $MODEFILE
systemctl daemon-reload
systemctl restart sm70-serve
for i in $(seq 1 180); do
  curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && { echo "HEALTH_OK after ~$((i*10))s"; break; }
  sleep 10
done
sleep 20
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)

echo "=== boot journal: AR hook state ==="
journalctl -u sm70-serve --since "2 min ago" --no-pager 2>/dev/null | grep -iE "hier|fp8_wire|topology" | head -8

measure_arm() {
  local arm="$1"
  echo "=== arm $arm: battery $(date) ==="
  ( cd /data/sm70_results/round5/ar_hier_ab && timeout 3600 /data/vllm-dev/bin/python \
      /data/src/vllm-v2/sm70_tests/check_online_content.py > arm_${arm}_battery.log 2>&1 )
  echo "BATTERY_RC=$?"
  tail -2 arm_${arm}_battery.log
  echo "=== arm $arm: ruler 65K $(date) ==="
  cd /data/src/vllm-v2
  timeout 900 /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 65536 \
      > $OUT/arm_${arm}_ruler.log 2>&1
  echo "RULER_RC=$?"
  grep -E "prefill=|baseline:" $OUT/arm_${arm}_ruler.log
}

echo "=== ARM 0 (NCCL path, mode=0) $(date) ==="
echo 0 > $MODEFILE; sleep 30
measure_arm 0

echo "=== flip to mode 1 $(date) ==="
echo 1 > $MODEFILE; sleep 30

echo "=== ARM 1 (hier route + fp8 wire, mode=1) $(date) ==="
measure_arm 1

# confirm arm if promising-looking is decided by the controller; second
# repeat of mode 1 for stability:
echo 1 > $MODEFILE; sleep 30
echo "=== ARM 1b (repeat) $(date) ==="
measure_arm 1b

echo "=== final: mode file left at 1; daemon tail check ==="
DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: daemon pids dead:$DAEMON_DEAD" || echo "daemon liveness OK (untouched)"
echo "=== A/B end $(date) ==="
exit 0
