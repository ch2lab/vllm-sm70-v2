#!/bin/bash
# Clean daemon+serve cycle for a measured boot.
#
# Why: a serve that is killed without a full teardown leaves the weight-cache
# IPC channel dirty. The next serve then logs
#   "Weight cache IPC loading failed; falling back to disk loading"
# rebuilds the model in the same process, and dies with
#   ValueError: Duplicate layer name: language_model.model.layers.0.mlp.experts
# (observed three times on 2026-10-24). Rebuilding the daemon each time keeps
# the IPC arm healthy and the comparison honest.
set -uo pipefail

echo "  [cycle] stopping serve"
systemctl stop sm70-serve 2>/dev/null
for i in $(seq 1 60); do
  systemctl is-active --quiet sm70-serve || break
  sleep 2
done
# Engine processes only; the daemon runs `...cli.main preload`, so match serve.
for i in $(seq 1 30); do
  pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null 2>&1 || break
  sleep 2
done

echo "  [cycle] stopping daemon"
systemctl stop sm70-daemon 2>/dev/null
for i in $(seq 1 60); do
  systemctl is-active --quiet sm70-daemon || break
  sleep 2
done
for i in $(seq 1 30); do
  pgrep -f "vllm.entrypoints.cli.main preload" >/dev/null 2>&1 || break
  sleep 2
done
sleep 10

echo "  [cycle] starting daemon"
DAEMON_START=$(date -Is)
systemctl start sm70-daemon
ready=0
for i in $(seq 1 90); do
  systemctl is-active --quiet sm70-daemon || { sleep 5; continue; }
  if journalctl -u sm70-daemon --since "$DAEMON_START" --no-pager 2>/dev/null \
       | grep -q "Weight cache daemon READY"; then
    ready=1; break
  fi
  sleep 15
done
echo "  [cycle] daemon ready=$ready"
bash /root/run/ram_guard.sh | tail -1

echo "  [cycle] starting serve"
systemctl start sm70-serve
