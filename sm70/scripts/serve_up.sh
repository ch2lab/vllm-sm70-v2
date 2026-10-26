#!/usr/bin/env bash
# serve_up.sh — restart ONLY the serve, then wait with the maintained detector.
# The weight-cache daemon must already be active (use sm70_up.sh for a cold start).
# Detection is NOT reimplemented here: serve_wait.sh reads the unit journal and
# distinguishes "Application startup complete" from real error lines.
#
# usage: serve_up.sh [timeout_s]   (default 1800; cold boot with JIT warmup is ~8-10 min)
set -uo pipefail
# One boot at a time (discipline 2026-10-19/23): the lock lives in /run (tmpfs).
exec 9>/run/sm70_boot.lock
flock -n 9 || { echo "another boot/experiment holds /run/sm70_boot.lock"; exit 3; }
TMO="${1:-1800}"
systemctl reset-failed sm70-serve 2>/dev/null
systemctl restart sm70-serve || { echo "FATAL: systemctl restart sm70-serve failed"; exit 1; }
exec /root/run/serve_wait.sh "$TMO" sm70-serve
