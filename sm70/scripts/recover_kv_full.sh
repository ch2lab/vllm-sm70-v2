#!/usr/bin/env bash
# Recover from the half-done KV migration: the previous daemon died leaving an
# orphan holding 8002, and the serve then fell back to disk (PLE _uva_weight bug).
# Start the daemon cleanly under the NEW key (full nvfp4) and only boot the serve
# once the weight cache is actually resident in GPU memory.
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/kvdtype-full-nvfp4
mkdir -p "$OUT"

echo "=== clean slate ==="
for pid in $(ps -eo pid,cmd | awk '/vllm.entrypoints.cli.main preload/ && !/awk/ {print $1}'); do
  echo "  kill preload pid $pid"; kill "$pid" 2>/dev/null || true
done
for pid in $(ss -ltnp 2>/dev/null | awk '/:8002 / {print $0}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  echo "  kill 8002 holder $pid"; kill "$pid" 2>/dev/null || true
done
systemctl stop sm70-serve 2>/dev/null || true
sleep 8
ss -ltnp 2>/dev/null | grep -q ":8002 " && { echo "  WARN 8002 still held"; } || echo "  8002 free"

echo "=== start daemon (full nvfp4 key) ==="
nohup bash /root/run/run_sm70_preload.sh >> "$OUT/preload2.log" 2>&1 &
echo "  daemon pid $!"
READY=0
for i in $(seq 1 90); do   # up to 15 min
  sleep 10
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | head -1)
  echo "  t=$((i*10))s min_gpu_used=${USED}MiB"
  if [ "${USED:-0}" -ge 15000 ]; then READY=1; echo "  weights resident"; break; fi
done
if [ "$READY" -ne 1 ]; then echo "  DAEMON NEVER RESIDENT"; tail -20 "$OUT/preload2.log"; exit 1; fi

echo "=== boot serve ==="
systemctl start sm70-serve
timeout 1800 bash /root/run/serve_wait.sh 1700 sm70-serve || { echo "SERVE FAILED"; exit 1; }

cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
echo "=== decode1 (retrying client) ==="
for attempt in 1 2 3; do
  timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
    --salt-seed 9503 --rounds 4 > "$OUT/decode1_final.log" 2>&1
  grep -q "^round=1 .*decode_agg=" "$OUT/decode1_final.log" && break
  sleep 20
done
grep -E "^round=" "$OUT/decode1_final.log" | grep -v warm | sed 's/^/  /'
SM70_RESULTS_DIR="$OUT" timeout 1200 $PY correctness.py 2>&1 | grep -E "prefix_cache|CORRECT-" | sed 's/^/  /'
timeout 900 $PY long_content_gate.py 2>&1 | tail -2 | sed 's/^/  /'
dmesg 2>/dev/null | grep -ci Xid | sed 's/^/  dmesg Xid lines: /'
