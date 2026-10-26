#!/usr/bin/env bash
# serve_wait.sh — IRON RULE (2026-10-19): never probe readiness via HTTP /health.
# Read the unit log in real time; on failure print the RAW tail (the actual
# exception line, not a wrapper) so the cause is visible immediately.
#
# Success marker : "Application startup complete"
# Failure        : unit enters failed state, or a terminal error line appears
#                  (ERROR / Traceback / RuntimeError / AttributeError / ...)
#
# usage: serve_wait.sh [timeout_s] [unit]
set -uo pipefail
TMO="${1:-900}"
UNIT="${2:-sm70-serve}"
START_TS=$(date '+%Y-%m-%d %H:%M:%S')

raw_tail() {
  echo "----- RAW LOG TAIL ($UNIT) -----"
  journalctl -u "$UNIT" --since "$START_TS" --no-pager 2>/dev/null | tail -n "${1:-60}"
  echo "----- systemctl status -----"
  systemctl status "$UNIT" --no-pager -l 2>&1 | head -12
}

deadline=$((SECONDS + TMO))
while [ "$SECONDS" -lt "$deadline" ]; do
  log=$(journalctl -u "$UNIT" --since "$START_TS" --no-pager 2>/dev/null)
  case "$UNIT" in
    *daemon*) MARK="Weight cache daemon READY" ;;
    *)        MARK="Application startup complete" ;;
  esac
  if printf '%s\n' "$log" | grep -q "$MARK"; then
    echo "[serve_wait] READY (log marker '$MARK')"
    exit 0
  fi
  # Daemon window caveat: `systemctl restart sm70-daemon` returns only after
  # ExecStartPost (wait_sm70_ple_ready.sh) finishes, which is AFTER the daemon
  # prints its READY marker. So the marker predates $START_TS and the scan above
  # never sees it (observed 2026-10-19: a 3h45m false wait while the daemon had
  # been ready for 15s). Re-scan from the main process start instead: that still
  # excludes stale markers from earlier invocations and stays log-based.
  case "$UNIT" in
    *daemon*)
      mstart=$(systemctl show -p ExecMainStartTimestamp --value "$UNIT" 2>/dev/null)
      if [ -n "$mstart" ]; then
        log2=$(journalctl -u "$UNIT" --since "$mstart" --no-pager 2>/dev/null)
        if printf '%s\n' "$log2" | grep -q "$MARK"; then
          echo "[serve_wait] READY (log marker '$MARK' since main-process start $mstart)"
          exit 0
        fi
      fi
      ;;
  esac
  # terminal error line = fail fast, with the real text
  # Only TERMINAL failures. Handled/recovered errors (e.g. "Weight cache IPC
  # loading failed; falling back to disk loading") are NOT terminal.
  term=$(printf '%s\n' "$log" | grep -E "Engine core initialization failed|Worker failed with error|WorkerProc failed to start|Traceback \(most recent call last\)" | tail -3)
  if printf '%s\n' "$log" | grep -q "Engine core initialization failed"; then term="$(printf '%s\n' "$log" | grep -E '^.*(RuntimeError|AttributeError|KeyError|AssertionError):' | tail -3)"; fi
  if systemctl is-failed --quiet "$UNIT" 2>/dev/null || [ -n "$term" ]; then
    echo "[serve_wait] FAILED"
    [ -n "$term" ] && { echo "----- terminal error line(s) -----"; printf '%s\n' "$term"; }
    raw_tail 80
    exit 1
  fi
  sleep 2
done
echo "[serve_wait] TIMEOUT after ${TMO}s (no success marker, no failure line)"
raw_tail 80
exit 2
