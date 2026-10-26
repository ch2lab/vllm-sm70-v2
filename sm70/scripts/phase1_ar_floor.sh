#!/usr/bin/env bash
# AR protocol-floor measurement (quiet window): in-graph, lockstep 8-rank timing
# of the hier route vs NCCL at the decode payload size (M=1 -> 5120 B).
# Decides whether the in-service 25.9 us/call is protocol or skew.
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/ar_protocol_floor
mkdir -p "$OUT"
PY=/data/vllm-dev/bin/python

systemctl stop sm70-serve
sleep 90
echo "=== window ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

cd /data/src/vllm-v2/sm70_tests
timeout 1500 $PY bench_ar_hier_gate1.py --verify-m 1 --rounds 300 --stress 0 \
  --no-skew --out "$OUT/gate1" > "$OUT/bench.log" 2>&1
RC=$?
echo "bench rc=$RC"
tail -40 "$OUT/bench.log" | sed 's/^/  /'

echo "=== restore serve ==="
/root/run/ram_guard.sh 30 ar_floor_restore || exit 3
systemctl start sm70-serve
timeout 1500 bash /root/run/serve_wait.sh 1400 sm70-serve || exit 1
