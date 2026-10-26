#!/bin/bash
# P6 G1 window (2026-10-15): in-engine A/B for the wide-160 MoE kernel.
# One window per arm: $1 = arm name, $2 = VLLM_SM70_MOE_QPN2_WIDE value
# (production baseline = "0"; candidate = "auto").
#
# Protocol = stepwall_window.sh (P5): setsid tree kill only, daemon
# pids read-only baseline + end recheck, cleanup trap restores the
# service on every exit path, window env mirrors the prod drop-in
# (SM70_QSA_FMA=1 + VLLM_PLE_QUANT_DIR via run script) so the only
# controlled diff between arms is the knob.
set -u
ARM=$1
WIDEVAL=$2
OUT=/data/sm70_results/round5/p6_moe/g1_$ARM
mkdir -p "$OUT"
cd "$OUT"

DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
echo "daemon pids (read-only baseline): ${DAEMON_PIDS:-NONE}"
[ -n "$DAEMON_PIDS" ] || { echo "FATAL: no daemon running before window — refuse to start"; exit 1; }

cleanup() {
  if ! systemctl is-active -q sm70-serve; then
    echo "cleanup: service not active -> restarting"
    systemctl start sm70-serve
  fi
}
trap cleanup EXIT

echo "=== window start $ARM wide=$WIDEVAL $(date) ==="
systemctl stop sm70-serve
sleep 5

export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
export SM70_QSA_FMA=1
export VLLM_SM70_MOE_QPN2_WIDE=$WIDEVAL
echo "env: SM70_QSA_FMA=1 VLLM_SM70_MOE_QPN2_WIDE=$WIDEVAL (matches prod drop-in + A/B knob)"

PGID_FILE=$OUT/pgid
rm -f "$PGID_FILE"
setsid bash -c "echo \$\$ > $PGID_FILE; exec /root/run/run_sm70_serve_ipc.sh" \
  > $OUT/boot.log 2>&1 &
for i in $(seq 1 20); do [ -s "$PGID_FILE" ] && break; sleep 0.5; done
SERVE_PID=$(cat "$PGID_FILE" 2>/dev/null)
echo "serve session leader (pgid) = $SERVE_PID"
[ -n "$SERVE_PID" ] || { echo "FATAL: serve tree not tracked"; exit 1; }

READY=0
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "READY=1 after ~$((i*10))s"
    READY=1
    break
  fi
  sleep 10
done
[ $READY -eq 1 ] || { echo "FATAL: not ready in 1800s"; kill -9 -- -"$SERVE_PID" 2>/dev/null; exit 1; }

# env propagation proof: the knob must be in every window-tree process's
# environ. Session-level (not pgrep-pattern): the vLLM workers are mp-spawned
# and their cmdline does NOT contain "vllm.entrypoints.cli.main", so a
# pattern check silently skips them (G1 lesson 2026-10-15).
rm -f /tmp/sm70_qpn2_moe_diag
for p in $(ps -e -o pid=,sess= | awk -v S="$SERVE_PID" '$2==S {print $1}'); do
  V=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep "VLLM_SM70_MOE_QPN2_WIDE" || true)
  echo "pid $p knob-env: ${V:-ABSENT}"
done
echo "=== content battery (6q x K=10) $(date) ==="
cd /data/src/vllm-v2
timeout 3600 /data/vllm-dev/bin/python sm70_tests/check_online_content.py \
  > $OUT/battery.log 2>&1
echo "BATTERY_RC=$?"
tail -15 $OUT/battery.log

echo "=== ruler 65K $(date) ==="
timeout 900 /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 65536 \
  > $OUT/ruler.log 2>&1
echo "RULER_RC=$?"
cat $OUT/ruler.log

echo "=== window stop $(date) ==="
kill -INT -- -"$SERVE_PID" 2>/dev/null
for i in $(seq 1 30); do
  kill -0 -- -"$SERVE_PID" 2>/dev/null || break
  sleep 10
done
kill -9 -- -"$SERVE_PID" 2>/dev/null
sleep 5
SRV_LEFT=$(pgrep -f "vllm.entrypoints.cli.main serve" || true)
[ -n "$SRV_LEFT" ] && echo "FATAL: stray serve pids: $SRV_LEFT" || echo "no stray serve processes"
DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: PROTOCOL VIOLATION — daemon pids dead:$DAEMON_DEAD" || echo "daemon liveness OK (untouched)"

echo "=== restart service $(date) ==="
systemctl start sm70-serve
OK=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "HEALTH_OK after ~$((i*10))s"
    OK=1
    break
  fi
  sleep 10
done
[ $OK -eq 1 ] || echo "FATAL: health not OK after 1200s"
echo "=== window end $ARM $(date) ==="
exit 0
