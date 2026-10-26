#!/usr/bin/env bash
# 10-17 Iter-7 (prefill axis first, user directive): prefill chunk 8192 -> 16384.
# Halves AR round-trips over a 65K prefill (8 -> 4 chunks) and gives the MoE
# GEMM a larger M. Single variable: --max-num-batched-tokens.
set -o pipefail
S=/root/run/run_sm70_serve_ipc.sh

echo "=== ITER-7 CHUNK16K $(date +%T) ==="
# wait for the in-flight production boot to be READY first
for i in $(seq 1 320); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] \
    && { echo "PROD READY ${i}x5s $(date +%T)"; break; }; sleep 5
done
echo "gpu-free-before: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"

/data/vllm-dev/bin/python - "$S" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
t = p.read_text()
old = "  --max-num-batched-tokens 8192 \\"
new = "  --max-num-batched-tokens 16384 \\"
assert old in t and "16384" not in t
p.write_text(t.replace(old, new))
print("chunk 8192 -> 16384")
PYEOF
bash -n $S
systemctl restart sm70-serve.service
for i in $(seq 1 320); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] \
    && { echo "C16K READY ${i}x5s $(date +%T)"; break; }; sleep 5
done
journalctl -u sm70-serve.service --no-pager --since "-30min" | grep -iE "error|oom|failed" | tail -3 | cut -c1-140
echo "gpu-free-after: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr '\n' ' ')"

cd /data/src/vllm-v2/sm70_tests
source /data/src/sm70_env.sh
echo "C16-prefill65K: $(timeout 300 /data/vllm-dev/bin/python qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 --rounds 2 2>&1 | tail -1)"
echo "C16-decode1: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 1 --decode-prefix-len 8192 --salt-seed 9502 --rounds 2 2>&1 | tail -1)"
echo "C16-conc8: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 8 --decode-prefix-len 8192 --salt-seed 9610 --rounds 2 2>&1 | tail -1)"
echo "C16-conc16: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 16 --decode-prefix-len 8192 --salt-seed 9611 --rounds 2 2>&1 | tail -1)"
echo "Xids: $(dmesg 2>/dev/null | grep -c Xid)"
