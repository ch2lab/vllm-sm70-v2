#!/usr/bin/env bash
# 10-17 Iter-6: MTP speculative decoding, single variable.
# Add --speculative-config to the serve script (MTP head ships in the NVFP4
# checkpoint; qwen4_exp -> qwen4_exp_mtp auto-derived, n_predict=1).
set -o pipefail
S=/root/run/run_sm70_serve_ipc.sh
OUT=/root/run/nsys
mkdir -p $OUT

echo "=== ITER-6 MTP $(date +%T) ==="
# ---- on: insert --speculative-config into the exec chain ----
/data/vllm-dev/bin/python - "$S" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
t = p.read_text()
FLAG = "  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 1}' \\\n"
if "speculative-config" in t:
    raise SystemExit("speculative-config already present")
anchor = "  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \\"
assert anchor in t, "anchor line not found"
t = t.replace(anchor, anchor + FLAG)
p.write_text(t)
print("speculative-config ON")
PYEOF
bash -n $S
systemctl restart sm70-serve.service
for i in $(seq 1 320); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] \
    && { echo "MTP READY ${i}x5s $(date +%T)"; break; }; sleep 5
done
journalctl -u sm70-serve.service --no-pager --since "-30min" | grep -i "spec\|MTP" | tail -4 | cut -c1-150

cd /data/src/vllm-v2/sm70_tests
source /data/src/sm70_env.sh
echo "MTP-prefill65K: $(timeout 300 /data/vllm-dev/bin/python qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 --rounds 2 2>&1 | tail -1)"
echo "MTP-decode1: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 1 --decode-prefix-len 8192 --salt-seed 9502 --rounds 2 2>&1 | tail -1)"
echo "MTP-conc8: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 8 --decode-prefix-len 8192 --salt-seed 9610 --rounds 2 2>&1 | tail -1)"
echo "MTP-conc16: $(timeout 400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 16 --decode-prefix-len 8192 --salt-seed 9611 --rounds 2 2>&1 | tail -1)"

# accuracy: greedy fixed prompt must match the wire=0 baseline text
PROMPT='请解释什么是张量并行(Tensor Parallelism),并用一句话说明它与流水并行的区别。'
curl -s --max-time 180 http://127.0.0.1:8001/v1/completions \
  -H "Authorization: Bearer ${SM70_API_KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen3.8-Flash-Next\",\"prompt\":\"$PROMPT\",\"temperature\":0,\"max_tokens\":32,\"n\":1}" \
  -o $OUT/greedy_mtp.json
echo "MTP greedy: $(head -c 400 $OUT/greedy_mtp.json)"
# spec accept rate from engine logs
journalctl -u sm70-serve.service --no-pager --since "-10min" | grep -i "accept" | tail -3 | cut -c1-200
echo "Xids: $(dmesg 2>/dev/null | grep -c Xid)"
