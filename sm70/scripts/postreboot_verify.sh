#!/usr/bin/env bash
# One-shot post-reboot verification (auto-installed before the 10-17 reboot
# to clear the GPU degraded state caused by the 10-16 20:30 Xid-13 storm).
# Sequence: wait GPU -> (no-op if serve already healthy) -> start preload
# -> wait 8002 -> start serve -> wait 8001 -> 65K ruler -> Xid delta report.
# Results: /root/run/postreboot_ruler.out and /root/run/postreboot_status.txt
set -uo pipefail

log() { echo "[$(date '+%F %T')] $*"; }
SELF=postreboot-verify.service

# no-op if the stack is already serving (protects against later reboots)
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ]; then
  log "serve already healthy - no-op"
  systemctl disable "$SELF" >/dev/null 2>&1
  exit 0
fi

# 1. wait for GPUs to come up
for i in $(seq 1 60); do
  nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1 && break
  sleep 5
done

# 2. start preload weight-cache daemon
log "starting preload"
nohup bash /root/run/run_sm70_preload.sh >> /root/run/ple_preload.log 2>&1 &

# 3. wait for preload health (8002), up to 15 min
ready=0
for i in $(seq 1 180); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8002/health 2>/dev/null)" = "200" ]; then ready=1; log "preload READY after $((i*5))s"; break; fi
  sleep 5
done
if [ "$ready" != "1" ]; then
  log "preload NOT ready after 15min"
  tail -20 /root/run/ple_preload.log > /root/run/postreboot_status.txt
  exit 1
fi

# 4. start serve
log "starting serve"
nohup bash /root/run/run_sm70_serve_ipc.sh > /tmp/prodab_serve.log 2>&1 &

# 5. wait for serve health (8001), up to 10 min
for i in $(seq 1 120); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ]; then log "serve READY after $((i*5))s"; break; fi
  sleep 5
done

# 6. 65K ruler (same ruler + seed as the baseline: lens 65536, salt 9502)
xid_before=$(dmesg 2>/dev/null | grep -c Xid || echo 0)
log "running 65K ruler"
cd /data/src/vllm-v2/sm70_tests && /data/vllm-dev/bin/python baseline.py --lens 65536 --salt-seed 9502 > /root/run/postreboot_ruler.out 2>&1
rc=$?

xid_after=$(dmesg 2>/dev/null | grep -c Xid || echo 0)
{
  echo "postreboot verification $(date '+%F %T')"
  echo "ruler_rc=$rc xid_before=$xid_before xid_after=$xid_after new_xids=$((xid_after - xid_before))"
  echo "=== ruler ==="
  cat /root/run/postreboot_ruler.out
} > /root/run/postreboot_status.txt

# 7. self-disable: this check must not repeat on future reboots
systemctl disable "$SELF" >/dev/null 2>&1
log "done (self-disabled); results in /root/run/postreboot_status.txt"
exit $rc
