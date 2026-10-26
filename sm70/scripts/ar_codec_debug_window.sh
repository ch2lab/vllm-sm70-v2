#!/bin/bash
# Quiet window: AR L1 codec debug unit (8 ranks, serve DOWN) + GDN K2 v2
# gates (G1 parity + G2 A/B). Both need exclusive GPUs; one window for both.
# Protocol: serve down, journal silent >= 60s, GPUs drained; daemon
# read-only baseline + tail liveness check.
set -uo pipefail
OUT=/data/sm70_results/round5/ar_codec_debug
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

# Daemon (PLE preload) holds ~18GB/GPU PERMANENTLY (untouchable) — the
# drain target is RELATIVE to the pre-stop baseline, not an absolute floor.
BASE_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
FLOOR=$((BASE_MEM + 500))
echo "=== stop serve + quiet wait $(date) (mem baseline ${BASE_MEM}MiB) ==="
systemctl stop sm70-serve
sleep 70  # journal-quiet >= 60s (G0 inside the unit re-verifies)
for i in $(seq 1 30); do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$MEM" -lt "$FLOOR" ] && { echo "GPUs drained to ${MEM}MiB (floor ${FLOOR}) after ~$((i*5))s"; break; }
  sleep 5
done

echo "=== AR codec debug unit $(date) ==="
timeout 1500 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/test_ar_codec_debug.py --out $OUT \
  > ar_codec_unit.log 2>&1
UNIT_RC=$?
echo "UNIT_RC=$UNIT_RC"
tail -25 ar_codec_unit.log

echo "=== GDN K2 v2 G1 parity $(date) ==="
timeout 3600 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/test_gdn_chunked_parity.py \
  > k2v2_g1_parity.log 2>&1
G1_RC=$?
echo "G1_RC=$G1_RC"
tail -6 k2v2_g1_parity.log

echo "=== GDN K2 v2 G2 A/B $(date) ==="
timeout 3600 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/bench_gdn_g2_ab.py \
  > k2v2_g2_ab.log 2>&1
G2_RC=$?
echo "G2_RC=$G2_RC"
tail -12 k2v2_g2_ab.log

echo "=== restore serve $(date) ==="
systemctl start sm70-serve
for i in $(seq 1 90); do
  curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && \
    { echo "HEALTH_OK ~$((i*10))s"; break; }
  sleep 10
done
sleep 20
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
timeout 1800 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/check_online_content.py > battery_restore.log 2>&1
echo "BATTERY_RC=$?"
tail -2 battery_restore.log

DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: daemon pids dead:$DAEMON_DEAD" \
  || echo "daemon liveness OK (untouched)"
echo "UNIT_RC=$UNIT_RC G1_RC=$G1_RC G2_RC=$G2_RC"
echo "=== quiet window end $(date) ==="
exit 0
