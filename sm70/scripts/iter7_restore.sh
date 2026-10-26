#!/bin/bash
# Iter-7 restore: revert diag print, chunk -> 8192, restart, full probe matrix
set -e
cd /data/src/vllm-v2
# 1. revert the QSA-DIAG print in indexer_qsa.py
/data/vllm-dev/bin/python - <<'PY'
from pathlib import Path
p = Path("vllm/models/qwen4_exp/nvidia/indexer_qsa.py")
t = p.read_text()
old = '''        elif out.shape != (num_tokens, self.packed_output_width):
            print(f"[QSA-DIAG] out={tuple(out.shape)} raw_n={num_tokens} "
                  f"width={self.packed_output_width} "
                  f"buf={tuple(attn.topk_indices_buffer.shape)}", flush=True)
            raise ValueError("QSA selection output has an invalid shape")'''
new = '''        elif out.shape != (num_tokens, self.packed_output_width):
            raise ValueError("QSA selection output has an invalid shape")'''
assert old in t, "diag print not found - check manually"
p.write_text(t.replace(old, new))
print("diag print reverted")
PY
# 2. chunk -> 8192
/data/vllm-dev/bin/python - <<'PY'
from pathlib import Path
p = Path("/root/run/run_sm70_serve_ipc.sh")
t = p.read_text()
assert "  --max-num-batched-tokens 16384 \\" in t, "chunk not 16384 - check manually"
p.write_text(t.replace("  --max-num-batched-tokens 16384 \\", "  --max-num-batched-tokens 8192 \\"))
print("chunk -> 8192")
PY
bash -n /root/run/run_sm70_serve_ipc.sh
git -C /data/src/vllm-v2 diff --stat
# 3. restart + wait
systemctl restart sm70-serve.service
for i in $(seq 1 320); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health 2>/dev/null)" = "200" ] && { echo "PROD READY ${i}x5s $(date +%T)"; break; }
  sleep 5
done
# 4. full probe matrix (production verification)
cd /data/src/vllm-v2/sm70_tests && source /data/src/sm70_env.sh
/data/vllm-dev/bin/python qsa_conc_tput.py --mode prefill --conc 1 --len 65536 --salt-seed 9502 --rounds 1 2>&1 | tail -1
/data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 1 --len 2048 --salt-seed 9502 --rounds 1 2>&1 | tail -1
/data/vllm-dev/bin/python qsa_conc_tput.py --mode decode --conc 8 --len 2048 --salt-seed 961001 --rounds 1 2>&1 | tail -1
echo "Xids: $(dmesg 2>/dev/null | grep -c Xid)"
echo "=== RESTORE DONE $(date +%T) ==="
