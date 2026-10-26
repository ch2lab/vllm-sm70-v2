#!/bin/bash
# Short quiet window: re-run the FIXED AR codec debug unit (v2 harness).
#
# Discipline (user ruling 2026-10-15, #11928): NO blind sleeps — every wait
# is a bounded condition poll; every phase has precondition + postcondition
# gates; any unexpected state ABORTS (cleanup stray children -> verify GPUs
# drained -> restore service -> verify health -> exit non-zero). No phase
# starts on a wounded predecessor.
set -uo pipefail
OUT=/data/sm70_results/round5/ar_codec_debug
mkdir -p $OUT
cd $OUT

# ---------------------------------------------------------------- utilities
log() { echo "[$(date +%H:%M:%S)] $*"; }
ABORTED=0
restore() {
  # idempotent: kill strays, start service if down, verify health (bounded)
  ABORTED=1
  local floor="${FLOOR:-999999}"
  pkill -f "test_ar_codec_debug.py" 2>/dev/null || true
  sleep 2  # process teardown grace (not a measurement wait)
  # verify GPUs drained to the daemon floor before anything else
  local i mem
  for i in $(seq 1 24); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$mem" -le "$floor" ] && break
    sleep 2
  done
  if ! systemctl is-active -q sm70-serve; then
    systemctl start sm70-serve
  fi
  for i in $(seq 1 90); do
    curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && { log "service healthy"; return 0; }
    sleep 5
  done
  log "RESTORE-FAIL: service not healthy after 7.5 min"
  return 1
}
abort() { log "ABORT: $*"; restore; exit 3; }
trap 'restore' EXIT

# ---------------------------------------------------------------- baseline
DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
[ -n "$DAEMON_PIDS" ] || { log "FATAL: no daemon"; exit 1; }
log "daemon pids (read-only baseline): $DAEMON_PIDS"
FLOOR=$(( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) + 500 ))
log "GPU mem floor ${FLOOR}MiB (pre-stop baseline +500)"

# ---------------------------------------------------------------- phase 0: stop + verify
systemctl stop sm70-serve
for i in $(seq 1 60); do
  systemctl is-active -q sm70-serve || break
  sleep 2
done
systemctl is-active -q sm70-serve && abort "service still active 120s after stop"
log "phase0a: service stopped"
# journal silent >= 60s: poll the age of the LAST journal line (condition, bounded)
for i in $(seq 1 40); do
  age=$(python3 - <<'PY'
import re, subprocess, sys
from datetime import datetime
try:
    out = subprocess.run(
        ["journalctl", "-u", "sm70-serve", "-n", "1", "--output=short-iso"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    ts = out.splitlines()[-1].split(" ", 1)[0]
    ts = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", ts)  # +0800 -> +08:00
    now = datetime.now().astimezone()
    ts_dt = datetime.fromisoformat(ts)
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=now.tzinfo)
    print(int((now - ts_dt).total_seconds()))
except Exception:
    print(-1)
PY
  )
  [ "${age:--1}" -ge 60 ] && break
  sleep 3
done
[ "${age:--1}" -ge 60 ] || abort "journal never reached 60s silence in 120s"
log "phase0b: journal silent ${age}s"
for i in $(seq 1 24); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$mem" -le "$FLOOR" ] && break
  sleep 2
done
[ "$mem" -le "$FLOOR" ] || abort "GPUs not drained to ${FLOOR} (still ${mem})"
log "phase0c: GPUs drained (max ${mem}MiB)"

# ---------------------------------------------------------------- phase 1: unit
log "phase1: AR codec debug unit (timeout 1200s)"
timeout 1200 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/test_ar_codec_debug.py --out "$OUT" \
  > ar_codec_unit.log 2>&1
UNIT_RC=$?
# postcondition: no strays, GPUs drained — regardless of rc
pkill -f "test_ar_codec_debug.py" 2>/dev/null || true
for i in $(seq 1 24); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$mem" -le "$FLOOR" ] && break
  sleep 2
done
[ "$mem" -le "$FLOOR" ] || abort "unit left GPUs undrained (max ${mem})"
tail -25 ar_codec_unit.log
log "phase1 done: UNIT_RC=$UNIT_RC (0=pass 2=mismatch-evidence 1=window/hang)"
[ "$UNIT_RC" -le 2 ] || abort "unit rc=$UNIT_RC (hang/fatal) — no further phases"

# ---------------------------------------------------------------- phase 2: restore + verify
systemctl is-active -q sm70-serve || systemctl start sm70-serve
for i in $(seq 1 90); do
  curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 || abort "service not healthy after restart"
sleep 10  # settle before battery (traffic warm-up, bounded)
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
timeout 1800 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/check_online_content.py > battery_restore.log 2>&1
BATTERY_RC=$?
tail -2 battery_restore.log
[ "$BATTERY_RC" -eq 0 ] || abort "post-window battery failed (rc=$BATTERY_RC)"

for p in $DAEMON_PIDS; do kill -0 "$p" 2>/dev/null || { log "FATAL: daemon pid $p dead"; ABORTED=1; }; done
log "daemon liveness $([ $ABORTED -eq 0 ] && echo OK || echo BREACHED)"
log "UNIT_RC=$UNIT_RC BATTERY_RC=$BATTERY_RC — window end"
[ "$ABORTED" -eq 0 ] && [ "$BATTERY_RC" -eq 0 ]
exit $?
