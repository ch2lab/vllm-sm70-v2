#!/usr/bin/env bash
# Noise-aware A/B protocol for the SM70 tree comparison.
#
# Why: one 65K decode round swings 17-27% on this box (measured 2026-10-22:
# 73.3 -> 93.2 tok/s on the same tree, same salt, consecutive rounds), which is the
# same size as the effect we are trying to resolve. So: fixed salts, many rounds,
# report median and min, and never compare single rounds.
#
# usage: ab_measure.sh <tag> [rounds_decode1] [rounds_conc8]
set -uo pipefail
TAG="${1:?tag}"
N1="${2:-6}"      # 1 warm + N1-1 timed
N8="${3:-4}"
OUT="/data/sm70_results/${TAG}"
mkdir -p "$OUT"
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python

pick() {  # median and min of the timed decode_agg values in a log
  /data/vllm-dev/bin/python - "$1" <<'PY'
import re, sys, statistics
vals = []
for line in open(sys.argv[1]).read().splitlines():
    if "mode=decode" in line and "warm" not in line:
        m = re.search(r"decode_agg=([0-9.]+)", line)
        if m:
            vals.append(float(m.group(1)))
if vals:
    print(f"n={len(vals)} median={statistics.median(vals):.1f} min={min(vals):.1f} max={max(vals):.1f} vals={vals}")
else:
    print("no timed rounds")
PY
}

echo "=== prefill 65K (salt 9502, conc1, 2 timed rounds) ==="
timeout 1200 $PY qsa_conc_tput.py --mode prefill --len 65536 --conc 1 \
  --salt-seed 9502 --rounds 3 > "$OUT/prefill_9502.log" 2>&1
grep -E "^round=" "$OUT/prefill_9502.log" | grep -v warm | sed 's/^/  /'

echo "=== decode1 65K (salt 9503, conc1, $((N1-1)) timed rounds) ==="
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds "$N1" > "$OUT/decode1_9503.log" 2>&1
grep -E "^round=" "$OUT/decode1_9503.log" | grep -v warm | sed 's/^/  /'
echo -n "  decode1 "; pick "$OUT/decode1_9503.log"

echo "=== conc8 65K (salt 9504, $((N8-1)) timed rounds) ==="
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 8 \
  --salt-seed 9504 --rounds "$N8" > "$OUT/conc8_9504.log" 2>&1
grep -E "^round=" "$OUT/conc8_9504.log" | grep -v warm | sed 's/^/  /'
echo -n "  conc8 "; pick "$OUT/conc8_9504.log"

echo "=== results in $OUT ==="
