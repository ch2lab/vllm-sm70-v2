#!/usr/bin/env bash
# Restore the last KNOWN-GOOD config (the transitional KV pair) so the box
# serves again. The full-nvfp4 migration needs the PLE disk-load path fixed
# (the weight stays meta there -> no UVA view); that is a code task.
# Everything via systemd units only.
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/kvdtype-full-nvfp4

echo "=== restore the 5 production scripts from the kvfix backups ==="
cp -a /tmp/opencode/run_sm70_serve_ipc.sh.bak_kvfix /root/run/run_sm70_serve_ipc.sh
cp -a /tmp/opencode/run_sm70_preload_sync.sh.bak_kvfix /root/run/run_sm70_preload_sync.sh
for f in run_sm70_preload.sh run_sm70_serve_sync_nomtp.sh run_sm70_serve_sync.sh; do
  b=$(ls -t "$OUT/${f}.bak_"* 2>/dev/null | head -1)
  if [ -n "${b:-}" ]; then
    cp -a "$b" "/root/run/$f"
    echo "  restored $f from $b"
  else
    echo "  WARN no backup for $f"
  fi
done
for f in run_sm70_serve_sync_nomtp.sh run_sm70_serve_sync.sh run_sm70_preload.sh run_sm70_serve_ipc.sh; do
  n=$(grep -cE 'kv-cache-dtype-(skip|override)-layers' "/root/run/$f")
  printf "  %-32s skip/override lines=%s\n" "$f" "$n"
done

echo "=== restart daemon (systemd, old key) then serve ==="
systemctl stop sm70-serve 2>/dev/null || true
systemctl reset-failed sm70-daemon sm70-serve 2>/dev/null || true
systemctl restart sm70-daemon
for i in $(seq 1 120); do
  sleep 10
  ST=$(systemctl is-active sm70-daemon 2>/dev/null)
  echo "  t=$((i*10))s daemon=$ST"
  [ "$ST" = "active" ] && break
  if [ "$ST" = "failed" ]; then
    echo "  DAEMON FAILED"; journalctl -u sm70-daemon -n 15 --no-pager | tail -15; exit 1
  fi
done
[ "$(systemctl is-active sm70-daemon)" = "active" ] || { echo "  DAEMON TIMEOUT"; exit 1; }

systemctl start sm70-serve
timeout 1800 bash /root/run/serve_wait.sh 1700 sm70-serve || { echo "SERVE FAILED"; exit 1; }

cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
for attempt in 1 2 3; do
  timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
    --salt-seed 9503 --rounds 3 > "$OUT/decode1_restored.log" 2>&1
  grep -q "^round=1 .*decode_agg=" "$OUT/decode1_restored.log" && break
  sleep 20
done
grep -E "^round=" "$OUT/decode1_restored.log" | grep -v warm | sed 's/^/  /'
