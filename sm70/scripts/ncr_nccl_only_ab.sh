#!/bin/bash
# A/B window (2026-10-21): ALL AR via the NCCL/PYNCCL dispatch chain (hier OFF)
# vs today's hier baseline.  Same ruler (salt 9502 family), same gates.
# Requires: unit already booted with VLLM_SM70_AR_HIER=0 and no HIER_FILE
# (arming proof is printed below; a "SM70 hier AR" line = NOT armed, abort).
set -uo pipefail
OUT=/data/sm70_results/ncr_nccl_only
mkdir -p $OUT
cd /data/src/vllm-v2 || exit 1
PY=/data/vllm-dev/bin/python
export SM70_API_KEY=$(grep -o 'SM70_API_KEY=[0-9a-f]\{32\}' /data/src/sm70_env.sh | head -1 | cut -d= -f2-)

log() { echo "[$(date '+%F %T')] $*"; }

XID0=$(dmesg 2>/dev/null | grep -c "Xid" || true)
log "xid baseline = $XID0"

# --- arming proof: hier must NOT have been built this boot ---
BOOT_TS=$(systemctl show -p ExecMainStartTimestamp --value sm70-serve)
journalctl -u sm70-serve --since "$BOOT_TS" --no-pager 2>/dev/null \
  | grep -E "SM70 hier AR|all-reduce backends \(in dispatch" | tail -4 > $OUT/arming.txt
cat $OUT/arming.txt
if grep -q "SM70 hier AR" $OUT/arming.txt; then
  log "FATAL: hier route was built this boot -> NOT the NCCL-only arm"; exit 1
fi
grep -q "Using \['PYNCCL'\]" $OUT/arming.txt || { log "FATAL: dispatch chain is not PYNCCL-only"; exit 1; }

bash /root/run/warmup.sh 900 sm70-serve > $OUT/warmup.log 2>&1
log "warmup rc=$? (tail)"; tail -2 $OUT/warmup.log

run() {  # name, args...
  local name=$1; shift
  log "== $name start"
  timeout 3600 $PY sm70_tests/qsa_conc_tput.py "$@" --rounds 3 2>&1 | tee $OUT/$name.log
  log "== $name rc=${PIPESTATUS[0]}"
}

run prefill65k --mode prefill --conc 1 --len 65536 --salt-seed 9502 --tag ncclonly_pf65k
run prefill8k  --mode prefill --conc 1 --len 8192  --salt-seed 9505 --tag ncclonly_pf8k
run decode1    --mode decode  --conc 1 --len 65536 --salt-seed 9509 --tag ncclonly_d1
run decode8    --mode decode  --conc 8 --len 65536 --salt-seed 9513 --tag ncclonly_d8

log "== canary (check_online_content, 9 x K=10)"
timeout 3600 $PY sm70_tests/check_online_content.py > $OUT/canary.log 2>&1
log "CANARY_RC=$?"; tail -3 $OUT/canary.log

log "== long-context content gate"
timeout 1800 $PY sm70_tests/long_content_gate.py > $OUT/content_gate.log 2>&1
log "CONTENT_GATE_RC=$?"; tail -3 $OUT/content_gate.log

XID1=$(dmesg 2>/dev/null | grep -c "Xid" || true)
log "xid after = $XID1 (delta $((XID1 - XID0)))"
dmesg 2>/dev/null | grep "Xid" | tail -5 > $OUT/xid.txt

log "== summary"
grep -h "^round=" $OUT/prefill65k.log $OUT/prefill8k.log $OUT/decode1.log $OUT/decode8.log \
  | grep -v warm > $OUT/summary.txt
cat $OUT/summary.txt
log "DONE"
