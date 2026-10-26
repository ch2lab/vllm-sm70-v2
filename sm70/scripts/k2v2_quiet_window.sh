#!/bin/bash
# GDN K2 v2 quiet-window validation (runs AFTER ar_hier window #3 ends).
# Protocol: quiet window (serve down, journal quiet >= 60s, GPUs idle) for
# offline A/B integrity; daemon read-only baseline + tail check.
#   G1 = test_gdn_chunked_parity.py   (rel_l2 vs FP64, T in {256,100,8192},
#        h0 non-zero) — correctness gate (1e-3 / 3e-4)
#   G2 = bench_gdn_g2_ab.py          (shadow/incumbent device-time ratio,
#        PASS <= 0.5, MARGINAL <= 0.75)
# Restore: serve back up + quick battery (re-anchor).
set -uo pipefail
OUT=/data/sm70_results/round5/p3_gdn/k2v2
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

echo "=== stop serve + quiet wait $(date) ==="
systemctl stop sm70-serve
# journal-quiet >= 60s + GPU drain (invariant)
sleep 60
for i in $(seq 1 30); do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$MEM" -lt 4000 ] && { echo "GPUs drained (max ${MEM}MiB) after ~$((i*5))s"; break; }
  sleep 5
done

echo "=== G1 parity $(date) ==="
timeout 3600 /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/test_gdn_chunked_parity.py \
  > g1_parity.log 2>&1
G1_RC=$?
echo "G1_RC=$G1_RC"
tail -6 g1_parity.log
if [ $G1_RC -ne 0 ]; then
  echo "G1 FAIL -> no G2, restoring service"
  systemctl start sm70-serve
  sleep 120
  echo "=== G1-FAIL end $(date) ==="
  exit 2
fi

echo "=== G2 A/B $(date) ==="
timeout 3600 /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/bench_gdn_g2_ab.py \
  > g2_ab.log 2>&1
G2_RC=$?
echo "G2_RC=$G2_RC"
tail -12 g2_ab.log

echo "=== restore serve $(date) ==="
systemctl start sm70-serve
for i in $(seq 1 90); do
  curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && { echo "HEALTH_OK ~$((i*10))s"; break; }
  sleep 10
done
sleep 20
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
( cd $OUT && timeout 1800 /data/vllm-dev/bin/python \
    /data/src/vllm-v2/sm70_tests/check_online_content.py > battery_restore.log 2>&1 )
echo "BATTERY_RC=$?"
tail -2 battery_restore.log

DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: daemon pids dead:$DAEMON_DEAD" \
  || echo "daemon liveness OK (untouched)"
echo "=== k2v2 quiet window end $(date) ==="
exit 0
