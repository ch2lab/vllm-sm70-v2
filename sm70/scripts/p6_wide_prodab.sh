#!/bin/bash
# P6 G1 production A/B (2026-10-15): flip the wide-256 knob via systemd
# drop-in env (file-level op) + normal service restarts. Same protocol
# discipline as the window: daemon pids read-only, rechecked at the end.
#
# Sequence: baseline restart (no knob) -> battery + ruler -> add drop-in
# -> restart (knob auto) -> activation proof (C++ diag + live environ)
# -> battery + ruler. If the knob does NOT activate in workers, the
# drop-in is removed and the service is restarted back to baseline.
set -u
OUT=/data/sm70_results/round5/p6_moe/g1_prod
mkdir -p $OUT
cd $OUT

DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
echo "daemon pids (read-only baseline): ${DAEMON_PIDS:-NONE}"
[ -n "$DAEMON_PIDS" ] || { echo "FATAL: no daemon — refuse"; exit 1; }

# battery/ruler clients authenticate with the key (file-level grep, never printed)
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)

wait_health() {
  for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
      echo "HEALTH_OK after ~$((i*10))s"; return 0
    fi
    sleep 10
  done
  echo "FATAL: health not OK in 1800s"; return 1
}

# --- arm 1: baseline (current drop-in, no knob) ---------------------------
echo "=== baseline restart $(date) ==="
rm -f /tmp/sm70_qpn2_moe_diag
systemctl restart sm70-serve
wait_health || exit 1
echo "--- baseline diag (boot warmup MoE calls) ---"
sleep 20
cat /tmp/sm70_qpn2_moe_diag 2>/dev/null | head -4
timeout 3600 /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/check_online_content.py > $OUT/battery_baseline.log 2>&1
echo "BATTERY_BASELINE_RC=$?"
tail -3 $OUT/battery_baseline.log
cd /data/src/vllm-v2
timeout 900 /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 65536 > $OUT/ruler_baseline.log 2>&1
echo "RULER_BASELINE_RC=$?"; grep -E "prefill=|baseline:" $OUT/ruler_baseline.log
cd $OUT

# --- arm 2: knob via drop-in ----------------------------------------------
echo "=== add drop-in $(date) ==="
cat > /etc/systemd/system/sm70-serve.service.d/sm70_qpn2_wide.conf <<'EOF'
[Service]
Environment=VLLM_SM70_MOE_QPN2_WIDE=auto
EOF
systemctl daemon-reload
echo "=== knob restart $(date) ==="
rm -f /tmp/sm70_qpn2_moe_diag
systemctl restart sm70-serve
wait_health || exit 1

echo "--- activation proof: C++ diag (live C environ + dispatch decision) ---"
sleep 20
cat /tmp/sm70_qpn2_moe_diag 2>/dev/null | head -10
ACTIVATE=0
grep -q "path=wide" /tmp/sm70_qpn2_moe_diag 2>/dev/null && ACTIVATE=1
echo "ACTIVATE=$ACTIVATE"

# live process env check (session-free: all VLLM:: worker pids)
for p in $(pgrep -f "VLLM::Worker_TP" || true); do
  V=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep "VLLM_SM70_MOE_QPN2_WIDE" || true)
  echo "worker $p exec-env: ${V:-ABSENT}"
done
WPID=$(pgrep -f "VLLM::Worker_TP0" | head -1)
if [ -n "$WPID" ]; then
  echo "worker live-env via /proc$WPID/environ is frozen-at-exec; C++ raw_wide_env above is authoritative"
fi

timeout 3600 /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/check_online_content.py > $OUT/battery_knob.log 2>&1
echo "BATTERY_KNOB_RC=$?"
tail -3 $OUT/battery_knob.log
cd /data/src/vllm-v2
timeout 900 /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 65536 > $OUT/ruler_knob.log 2>&1
echo "RULER_KNOB_RC=$?"; grep -E "prefill=|baseline:" $OUT/ruler_knob.log
cd $OUT

if [ $ACTIVATE -ne 1 ]; then
  echo "=== knob did not activate in workers -> removing drop-in, restoring baseline ==="
  rm -f /etc/systemd/system/sm70-serve.service.d/sm70_qpn2_wide.conf
  systemctl daemon-reload
  systemctl restart sm70-serve
  wait_health || exit 1
else
  echo "=== knob active -> leaving it on in production ==="
fi

DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: PROTOCOL VIOLATION — daemon pids dead:$DAEMON_DEAD" || echo "daemon liveness OK (untouched)"
echo "=== prod A/B done $(date) ==="
exit 0
