#!/usr/bin/env bash
# Iter-8 gating A/B: graph vs eager on the CONC8 axis (single var:
# --cudagraph-capture-sizes "1 2 4 8 16 32 64" -> "1").
# bs=1 decode1 -> size-1 graph (UNCHANGED vs production)
# bs=8/16 decode -> 8/16 > max-capture 1 -> EAGER (test)
# prefill (2048/8192/65536 chunks) -> eager (UNCHANGED path vs production: >64)
# Same-day graph control: the iter8 EAGER-AB GRAPH window (13:36) + re-verify below.
set -uo pipefail
PY=/data/vllm-dev/bin/python
echo "=== ITER8 CONC8-EAGER-AB $(date +%T) ==="

# --- CAPTURE-1 (conc8/16 eager) window ---
systemctl stop sm70-serve.service
$PY - <<'EOF'
from pathlib import Path
p = Path("/root/run/run_sm70_serve_ipc.sh")
t = p.read_text()
assert "  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \\" in t, "capture line changed - check"
t = t.replace("  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \\",
              "  --cudagraph-capture-sizes 1 \\")
p.write_text(t)
print("serve -> capture-sizes 1 (conc8/16 eager)")
EOF
systemctl start sm70-serve.service
for i in $(seq 1 96); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] && { echo "SERVE-CAP1 READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
grep -n "cudagraph-capture-sizes" /root/run/run_sm70_serve_ipc.sh
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
echo "=== CAP1 MATRIX $(date +%T) ==="
$PY qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 1 --len 2048 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 8 --len 2048 --salt-seed 961001 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 16 --len 2048 --salt-seed 961002 2>&1 | tail -2
echo "Xids-cap1: $(dmesg 2>/dev/null | grep -c Xid)"

# --- RESTORE (production capture sizes) + re-verify ---
systemctl stop sm70-serve.service
$PY - <<'EOF'
from pathlib import Path
p = Path("/root/run/run_sm70_serve_ipc.sh")
t = p.read_text()
assert "  --cudagraph-capture-sizes 1 \\" in t, "capture-1 line not found - check"
t = t.replace("  --cudagraph-capture-sizes 1 \\",
              "  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \\")
p.write_text(t)
print("serve -> production capture sizes (restored)")
EOF
systemctl start sm70-serve.service
for i in $(seq 1 96); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] && { echo "SERVE-PROD READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
echo "=== RESTORE RE-VERIFY (graph decode1 + conc8) $(date +%T) ==="
$PY qsa_conc_tput.py --mode decode --conc 1 --len 2048 --salt-seed 9502 2>&1 | tail -2
$PY qsa_conc_tput.py --mode decode --conc 8 --len 2048 --salt-seed 961001 2>&1 | tail -2
echo "Xids-final: $(dmesg 2>/dev/null | grep -c Xid)"
echo "=== ITER8 CONC8-EAGER-AB DONE $(date +%T) ==="
