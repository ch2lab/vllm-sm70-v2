#!/usr/bin/env bash
# Iter-8 A/B: decode CUDA-graph vs eager (single var --enforce-eager on serve).
# Same-day drift control: EAGER window then GRAPH window, full matrix each.
set -uo pipefail
PY=/data/vllm-dev/bin/python
SVC=/root/run/run_sm70_serve_ipc.sh
echo "=== ITER8 EAGER-AB $(date +%T) ==="

# --- EAGER window ---
systemctl stop sm70-serve.service
$PY - <<'EOF'
from pathlib import Path
p = Path("/root/run/run_sm70_serve_ipc.sh")
t = p.read_text()
assert "--enforce-eager" not in t, "already eager?"
t = t.replace("  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \\",
              "  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \\\n  --enforce-eager \\")
p.write_text(t)
print("serve -> enforce-eager")
EOF
systemctl start sm70-serve.service
for i in $(seq 1 96); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] && { echo "SERVE-EAGER READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
echo "=== EAGER MATRIX $(date +%T) ==="
$PY qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 1 --len 2048 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 8 --len 2048 --salt-seed 961001 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 16 --len 2048 --salt-seed 961002 2>&1 | tail -2
echo "Xids-eager: $(dmesg 2>/dev/null | grep -c Xid)"

# --- GRAPH window (drift control) ---
systemctl stop sm70-serve.service
$PY - <<'EOF'
from pathlib import Path
p = Path("/root/run/run_sm70_serve_ipc.sh")
t = p.read_text()
assert "  --enforce-eager \\" in t, "eager flag not found - check"
p.write_text(t.replace("  --enforce-eager \\\n", ""))
print("serve -> graph (restored)")
EOF
systemctl start sm70-serve.service
for i in $(seq 1 96); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] && { echo "SERVE-GRAPH READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
echo "=== GRAPH MATRIX $(date +%T) ==="
$PY qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 1 --len 2048 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 8 --len 2048 --salt-seed 961001 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 16 --len 2048 --salt-seed 961002 2>&1 | tail -2
echo "Xids-graph: $(dmesg 2>/dev/null | grep -c Xid)"
echo "=== ITER8 EAGER-AB DONE $(date +%T) ==="
