#!/usr/bin/env bash
# One-shot after the 2026-10-24 Xid-154 reboot: bring the baseline back.
# The serve unit is disabled, and the agent shell does not survive a reboot,
# so this restores the known-good state exactly once and then disables itself.
set -uo pipefail
exec >>/data/sm70_results/postreboot_serve.log 2>&1
echo "=== $(date -Is) post-reboot bring-up ==="
for i in $(seq 1 60); do
  if systemctl is-active --quiet sm70-daemon; then break; fi
  sleep 10
done
sleep 30
bash /root/run/ram_guard.sh || { echo "ram_guard FAILED"; exit 1; }
systemctl start sm70-serve || { echo "serve start FAILED"; exit 1; }
if /root/run/serve_wait.sh 1500; then
  echo "serve READY"
  cd /data/src/vllm-v2/sm70_tests && timeout 900 /data/vllm-dev/bin/python correctness.py
  echo "canary rc=$?"
else
  echo "serve NOT READY (see journal)"
fi
systemctl disable sm70-postboot-once.service
echo "=== $(date -Is) done ==="
