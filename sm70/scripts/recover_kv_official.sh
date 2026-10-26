#!/usr/bin/env bash
# Recover via the OFFICIAL path: systemd daemon unit (ExecStartPost = the
# 8002/health readiness gate) then the serve unit. My earlier nohup daemon held
# 8002 and blocked sm70-daemon.service.
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/kvdtype-full-nvfp4
mkdir -p "$OUT"

echo "=== stop my manual daemon (holds 8002) ==="
for pid in $(ps -eo pid,cmd | awk '/vllm.entrypoints.cli.main preload/ && !/awk/ {print $1}'); do
  echo "  kill $pid"; kill "$pid" 2>/dev/null || true
done
sleep 6
systemctl reset-failed sm70-daemon.service 2>/dev/null || true
ss -ltn 2>/dev/null | grep -q ":8002 " && echo "  WARN: 8002 still held" || echo "  8002 free"

echo "=== start daemon unit (logs + health gate) ==="
systemctl start sm70-daemon.service &
for i in $(seq 1 120); do   # 20 min bound
  sleep 10
  ST=$(systemctl is-active sm70-daemon.service 2>/dev/null)
  echo "  t=$((i*10))s daemon=$ST"
  [ "$ST" = "active" ] && break
  [ "$ST" = "failed" ] && { echo "  DAEMON FAILED"; journalctl -u sm70-daemon -n 20 --no-pager | tail -20; exit 1; }
done
[ "$(systemctl is-active sm70-daemon.service)" = "active" ] || { echo "  DAEMON TIMEOUT"; exit 1; }
echo "  daemon READY (health gate passed)"

echo "=== boot serve (log marker wait) ==="
systemctl start sm70-serve
timeout 1800 bash /root/run/serve_wait.sh 1700 sm70-serve || { echo "SERVE FAILED"; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2 | sed 's/^/  /'

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
