#!/usr/bin/env bash
# Iter-7 ROLLBACK: 16K showed no meaningful gain (all axes within noise) + 0.9GB/GPU
# activation cost + daemon-consistency fragility -> revert serve+daemon to 8192.
set -uo pipefail
PY=/data/vllm-dev/bin/python
echo "=== ITER7 ROLLBACK 8K $(date +%T) ==="
systemctl stop sm70-serve.service
echo "serve stopped $(date +%T)"
$PY - <<'EOF'
from pathlib import Path
p = Path("/root/run/run_sm70_preload.sh")
t = p.read_text()
assert "  --max-num-batched-tokens 16384 \\" in t, "daemon not 16384 - check"
p.write_text(t.replace("  --max-num-batched-tokens 16384 \\", "  --max-num-batched-tokens 8192 \\"))
print("daemon -> 8192")
EOF
systemctl restart sm70-daemon.service
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8002/health 2>/dev/null)" = "200" ] && { echo "DAEMON-8192 READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
$PY - <<'EOF'
from pathlib import Path
p = Path("/root/run/run_sm70_serve_ipc.sh")
t = p.read_text()
assert "  --max-num-batched-tokens 16384 \\" in t, "serve not 16384 - check"
p.write_text(t.replace("  --max-num-batched-tokens 16384 \\", "  --max-num-batched-tokens 8192 \\"))
print("serve -> 8192")
EOF
systemctl start sm70-serve.service
for i in $(seq 1 320); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] && { echo "SERVE-8192 READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
echo "=== VERIFY 8192 MATRIX $(date +%T) ==="
$PY qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 --rounds 2 2>&1 | tail -1
$PY qsa_conc_tput.py --mode decode --conc 1 --len 2048 --salt-seed 9502 --rounds 2 2>&1 | tail -1
$PY qsa_conc_tput.py --mode decode --conc 8 --len 2048 --salt-seed 961001 --rounds 2 2>&1 | tail -1
echo "Xids: $(dmesg 2>/dev/null | grep -c Xid)"
echo "=== ROLLBACK DONE $(date +%T) ==="
