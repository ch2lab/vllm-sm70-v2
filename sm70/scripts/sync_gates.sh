#!/usr/bin/env bash
# Four-gate battery for a swapped SM70 tree: Xid, 65K ruler (salt 9502), decode1,
# conc8, MTP acceptance/step time, plus the content gates. Reads the server's own
# numbers (client wall time is the KPI; the per-request header is the cross-check).
#
# usage: sync_gates.sh [tag]        e.g. sync_gates.sh sync-main-2026-10-22
set -uo pipefail
TAG="${1:-$(date +%m%d-%H%M)}"
OUT="/data/sm70_results/${TAG}"
mkdir -p "$OUT"
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python

echo "=== [0] Xid since boot ==="
dmesg 2>/dev/null | grep -ci "Xid" | sed 's/^/  dmesg Xid lines: /'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader |
  sed 's/^/  gpu: /'

echo "=== [1] 65K ruler prefill (salt 9502, conc 1) ==="
timeout 1800 $PY qsa_conc_tput.py --mode prefill --len 65536 --conc 1 \
  --salt-seed 9502 --rounds 2 2>&1 | tee "$OUT/prefill_65k_9502.log" |
  grep -E "^round=|hdr|prefill_agg" | tail -4 | sed 's/^/  /'

echo "=== [2] decode1 (65K ctx, fresh salt) ==="
SALT=$((RANDOM + 70000))
timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed "$SALT" --rounds 2 2>&1 | tee "$OUT/decode1_${SALT}.log" |
  grep -E "^round=" | tail -2 | sed 's/^/  /'

echo "=== [3] conc8 (65K ctx, fresh salt) ==="
SALT8=$((RANDOM + 80000))
timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 8 \
  --salt-seed "$SALT8" --rounds 2 2>&1 | tee "$OUT/conc8_${SALT8}.log" |
  grep -E "^round=" | tail -2 | sed 's/^/  /'

echo "=== [4] MTP acceptance / step time ==="
/tmp/opencode/mtp_metric.sh $((RANDOM + 90000)) 2>&1 | sed 's/^/  /'

echo "=== [5] content gates ==="
for gate in long_content_gate.py correctness.py; do
  [ -f "$gate" ] || continue
  echo "  --- $gate"
  timeout 1800 $PY "$gate" 2>&1 | tail -6 | sed 's/^/    /'
done
echo "=== results in $OUT ==="
