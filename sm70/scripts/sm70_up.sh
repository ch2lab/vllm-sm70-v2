#!/usr/bin/env bash
# sm70_up.sh — canonical cold bring-up for the SM70 stack: ram guard -> weight-cache
# daemon (blocking on its readiness gate) -> serve -> log-based wait.
#
# Single entry point on purpose: detection lives in serve_wait.sh (journal marker
# "Application startup complete" vs real error lines / unit failure); NEVER probe
# HTTP /health. Do not write per-boot variants of this script.
#
# usage: sm70_up.sh            # log -> /root/run/sm70_up.log
set -uo pipefail
# One boot at a time (discipline 2026-10-19/23): the lock lives in /run (tmpfs).
exec 9>/run/sm70_boot.lock
flock -n 9 || { echo "another boot/experiment holds /run/sm70_boot.lock"; exit 3; }
LOG="${SM70_UP_LOG:-/root/run/sm70_up.log}"
exec > >(tee -a "$LOG") 2>&1
ts() { date '+%F %T'; }

echo "=== $(ts) sm70_up start ==="
/root/run/ram_guard.sh
echo "ram_guard rc=$?"
systemctl reset-failed sm70-daemon sm70-serve 2>/dev/null

if ! systemctl is-active --quiet sm70-daemon; then
  echo "$(ts) starting sm70-daemon (blocks until the unit's readiness gate passes)"
  if ! systemctl start sm70-daemon; then
    echo "$(ts) daemon start FAILED"
    journalctl -u sm70-daemon -n 40 --no-pager
    exit 1
  fi
  echo "$(ts) daemon active"
fi

echo "$(ts) starting sm70-serve (queued; Requires=sm70-daemon is already satisfied)"
systemctl start sm70-serve --no-block
/root/run/serve_wait.sh 1800 sm70-serve
rc=$?
echo "$(ts) serve_wait rc=$rc"

pid=$(systemctl show -p MainPID --value sm70-serve)
if [ -n "$pid" ] && [ "$pid" != "0" ] && [ -r "/proc/$pid/cmdline" ]; then
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
  echo "MainPID=$pid"
  echo "argv: $(echo "$cmd" | grep -o -- '--max-num-seqs [0-9]*')"
  echo "argv: $(echo "$cmd" | grep -o -- '--cudagraph-capture-sizes [0-9 ]*' | head -1)"
  echo "argv: $(echo "$cmd" | grep -o -- '--speculative-config [^ ]*')"
  echo "argv: $(echo "$cmd" | grep -o -- '--kv-cache-dtype [a-z0-9_]*')"
  echo "skip-layers-in-argv=$(echo "$cmd" | grep -c 'skip-layers')"
else
  echo "MainPID missing"
fi
echo "=== $(ts) sm70_up done rc=$rc ==="
exit "$rc"
