#!/usr/bin/env bash
# 10-17 Iter-5: AR wire codec single-variable test, wire 0 -> 2 (nvfp4).
# Stage 1: boot wire=0 (verified production), grab greedy baseline text.
# Stage 2: switch to wire=2, restart, full KPI matrix + greedy compare.
# Decision later: keep if KPI gain >= 3% AND greedy text within tolerance.

set -o pipefail
S=/root/run/run_sm70_serve_ipc.sh
OUT=/root/run/nsys
mkdir -p $OUT

echo "=== ITER-5 W2 $(date +%T) ==="
echo "script-before: $(grep -n 'AR_WIRE' $S)"

# ---------- Stage 1: production wire=0 boot + baseline text ----------
/data/vllm-dev/bin/python - "$S" <<'PYEOF'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
t = p.read_text()
t2, n = re.subn(r"export VLLM_SM70_AR_WIRE=\d+[^\n]*",
                "export VLLM_SM70_AR_WIRE=0", t)
assert n == 1, f"expected exactly 1 AR_WIRE line, got {n}"
p.write_text(t2)
print("AR_WIRE -> 0 (production baseline)")
PYEOF
bash -n $S
systemctl restart sm70-serve.service
for i in $(seq 1 288); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] \
    && { echo "W0 READY ${i}x5s $(date +%T)"; break; }; sleep 5
done
journalctl -u sm70-serve.service --no-pager | grep "hier AR: route ACTIVE" | tail -1 | cut -c1-160

source /data/src/sm70_env.sh
PROMPT='请解释什么是张量并行(Tensor Parallelism),并用一句话说明它与流水并行的区别。'
curl -s --max-time 180 http://127.0.0.1:8001/v1/completions \
  -H "Authorization: Bearer ${SM70_API_KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen3.8-Flash-Next\",\"prompt\":\"$PROMPT\",\"temperature\":0,\"max_tokens\":32,\"n\":1}" \
  -o $OUT/greedy_wire0.json
echo "W0 baseline: $(head -c 400 $OUT/greedy_wire0.json)"

# ---------- Stage 2: wire=2 (nvfp4) restart + KPI matrix ----------
/data/vllm-dev/bin/python - "$S" <<'PYEOF'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
t = p.read_text()
t2, n = re.subn(r"export VLLM_SM70_AR_WIRE=\d+[^\n]*",
                "export VLLM_SM70_AR_WIRE=2  # 10-17 Iter-5: nvfp4 wire (e2m1 + per-16 e4m3); rollback = 0 + restart", t)
assert n == 1
p.write_text(t2)
print("AR_WIRE -> 2 (nvfp4 wire)")
PYEOF
bash -n $S
systemctl restart sm70-serve.service
for i in $(seq 1 288); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] \
    && { echo "W2 READY ${i}x5s $(date +%T)"; break; }; sleep 5
done
journalctl -u sm70-serve.service --no-pager | grep "hier AR: route ACTIVE" | tail -1 | cut -c1-160

cd /data/src/vllm-v2/sm70_tests
echo "W2-prefill65K: $(timeout 300 /data/vllm-dev/bin/python qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 --rounds 2 2>&1 | tail -1)"
echo "W2-decode1: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 1 --decode-prefix-len 8192 --salt-seed 9502 --rounds 2 2>&1 | tail -1)"
echo "W2-conc8: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 8 --decode-prefix-len 8192 --salt-seed 9610 --rounds 2 2>&1 | tail -1)"
echo "W2-conc16: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 16 --decode-prefix-len 8192 --salt-seed 9611 --rounds 2 2>&1 | tail -1)"

curl -s --max-time 180 http://127.0.0.1:8001/v1/completions \
  -H "Authorization: Bearer ${SM70_API_KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen3.8-Flash-Next\",\"prompt\":\"$PROMPT\",\"temperature\":0,\"max_tokens\":32,\"n\":1}" \
  -o $OUT/greedy_wire2.json
echo "W2 greedy: $(head -c 400 $OUT/greedy_wire2.json)"
echo "Xids: $(dmesg 2>/dev/null | grep -c Xid)"
