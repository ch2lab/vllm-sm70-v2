#!/bin/bash
# A/B window 2026-10-21: VLLM_SM70_AR_HIER_MAX_BYTES arm measurement.
# usage: ar_cap16k_measure.sh <arm-label> [skip-warmup]
# Ruler: 65K salt 9502 family (qsa_conc_tput), 4 points x 2 timed rounds
# (--rounds 3 = 1 warm + 2 timed). Gates: canary + content + Xid delta.
set -uo pipefail
ARM="${1:?arm label}"
OUT=/data/sm70_results/ncr_cap16k/$ARM
mkdir -p "$OUT"
cd /data/src/vllm-v2 || exit 1
PY=/data/vllm-dev/bin/python
export SM70_API_KEY=$(grep -o 'SM70_API_KEY=[0-9a-f]\{32\}' /data/src/sm70_env.sh | head -1 | cut -d= -f2-)

log() { echo "[$(date '+%F %T')] $*"; }
log "ARM=$ARM OUT=$OUT"

# --- arming proof ---
S=$(systemctl show -p ExecMainStartTimestamp --value sm70-serve)
journalctl -u sm70-serve --since "$S" --no-pager 2>/dev/null \
  | grep -E "SM70 hier AR: route ACTIVE|SM70 hier AR: buffers sized|VLLM_SM70_AR_HIER_MAX_BYTES|all-reduce backends" | head -12 > $OUT/arming.txt
systemctl show -p Environment sm70-serve > $OUT/env.txt
cat $OUT/arming.txt; cat $OUT/env.txt

XID0=$(dmesg 2>/dev/null | grep -c "Xid")
log "xid baseline = $XID0"

if [ "${2:-}" != "skip-warmup" ]; then
  bash /root/run/warmup.sh 900 sm70-serve > $OUT/warmup.log 2>&1
  log "warmup rc=$?"; tail -2 $OUT/warmup.log
fi

run() {  # name, args...
  local name=$1; shift
  log "== $name start"
  timeout 3600 $PY sm70_tests/qsa_conc_tput.py "$@" --rounds 3 2>&1 | tee $OUT/$name.log
  log "== $name rc=${PIPESTATUS[0]}"
}

run prefill65k --mode prefill --conc 1 --len 65536 --salt-seed 9502 --tag ${ARM}_pf65k
run prefill8k  --mode prefill --conc 1 --len 8192  --salt-seed 9505 --tag ${ARM}_pf8k
run decode1    --mode decode  --conc 1 --len 65536 --salt-seed 9509 --tag ${ARM}_d1
run decode8    --mode decode  --conc 8 --len 65536 --salt-seed 9513 --tag ${ARM}_d8

log "== canary (check_online_content)"
timeout 3600 $PY sm70_tests/check_online_content.py > $OUT/canary.log 2>&1
log "CANARY_RC=$?"; tail -12 $OUT/canary.log

log "== long-context content gate"
timeout 1800 $PY sm70_tests/long_content_gate.py > $OUT/content_gate.log 2>&1
log "CONTENT_GATE_RC=$?"; tail -5 $OUT/content_gate.log

XID1=$(dmesg 2>/dev/null | grep -c "Xid")
log "xid after = $XID1 (delta $((XID1 - XID0)))"
dmesg 2>/dev/null | grep "Xid" | tail -5 > $OUT/xid.txt

journalctl -u sm70-serve --since "$S" --no-pager 2>/dev/null \
  | grep -E "SM70 hier AR TRACE" > $OUT/hier_trace.txt
log "hier trace lines: $(wc -l < $OUT/hier_trace.txt)"
grep -oE "TRACE: (TAKE|SKIP-CAP) n=[0-9]+ B cap=[0-9]+ B" $OUT/hier_trace.txt | sort | uniq -c

log "== summary"
grep -h "^round=[1-9]" $OUT/prefill65k.log $OUT/prefill8k.log $OUT/decode1.log $OUT/decode8.log \
  | tee $OUT/summary.txt
log "DONE $ARM"
