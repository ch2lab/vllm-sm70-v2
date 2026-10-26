#!/bin/bash
# Safe serve restart: full teardown before start.
#
# `systemctl restart` returns after SIGTERM but the control group can still be
# dying ("Failed to kill control group"), and a new engine started into that
# window collides with the old one and dies with
#   ValueError: Duplicate layer name: language_model...
# (observed twice on 2026-10-24). This stops, waits for the unit to be
# inactive AND for engine processes to disappear, then starts.
set -uo pipefail
WAIT_DEAD=${1:-120}

systemctl stop sm70-serve 2>/dev/null
for i in $(seq 1 $((WAIT_DEAD / 2))); do
  systemctl is-active --quiet sm70-serve || break
  sleep 2
done
for i in $(seq 1 30); do
  pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null 2>&1 || break
  sleep 2
done
sleep 10
left=$(pgrep -cf "vllm.entrypoints.cli.main serve" 2>/dev/null || true)
echo "  teardown: active=$(systemctl is-active sm70-serve) leftover_engine_procs=${left:-0}"
systemctl start sm70-serve
