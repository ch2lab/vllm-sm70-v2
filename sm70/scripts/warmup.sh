#!/usr/bin/env bash
# SM70 boot 暖机：固化"首次大 M 图捕获"与"长上下文工作区"尖峰，
# 避免首个线上请求踩尖峰（实测首跑 conc32 会把 min_free 压到 ~169 MiB）。
# 用法: bash /root/run/warmup.sh [timeout] [unit]
set -uo pipefail
cd /data/src/vllm-v2
PY=/data/vllm-dev/bin/python
TIMEOUT="${1:-900}"; UNIT="${2:-sm70-serve}"
export SM70_API_KEY=$(grep -o 'SM70_API_KEY=[^ ]*' /data/src/sm70_env.sh | head -1 | cut -d= -f2-)

# 注意：serve_wait.sh 等的是"调用之后新出现的 READY marker"，服务已就绪时会一直等到超时 ✗
# ⇒ 暖机这里改用"近期日志已出现 marker"判定
# Poll the journal directly: serve_wait only matches markers that appear AFTER it
# starts, so calling it on an already-ready service hangs until timeout.
ready=0
for _ in $(seq 1 $(( ${TIMEOUT:-900} / 5 ))); do
  S0=$(systemctl show -p ExecMainStartTimestamp --value "$UNIT")
  if journalctl -u "$UNIT" --since "$S0" --no-pager 2>/dev/null | grep -q "Application startup complete"; then
    ready=1; break
  fi
  sleep 5
done
if [ "$ready" != "1" ]; then
  echo "warmup: 服务未 READY ✗ (journal 里没有 Application startup complete)"; exit 1
fi
F0=$(timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1); F0=${F0:-n/a}
echo "warmup: READY ✓ free=${F0} MiB"
timeout 600 $PY sm70_tests/qsa_conc_tput.py --mode decode  --conc 30 --len 65536 --rounds 1 --salt-seed 9502 --tag wu_c32 >/dev/null 2>&1
echo "warmup: ① 大 M + 长上下文 decode 完成（最坏形状：conc30@65K（=KV 容量上限，避开过订抖动））"
timeout 900 $PY sm70_tests/qsa_conc_tput.py --mode prefill --conc 1  --len 65536 --rounds 1 --salt-seed 9502 --tag wu_pf  >/dev/null 2>&1
echo "warmup: ② 65K prefill 完成"
  # ③ Long-context buckets: Triton specialisations are keyed on scalar shape
  # properties, so a 130K/256K prompt compiled its kernels *during inference*
  # (measured 2026-10-21: a 130K request paid ~14 s of JIT, 33.3 s vs the ~19 s
  # its neighbours imply). Compile them here, before the service is handed over.
  if [ "${WARMUP_LONGCTX:-1}" = "1" ]; then
    for L in 131072 262144; do
      timeout 900 $PY sm70_tests/qsa_conc_tput.py --mode prefill --conc 1 --len $L \
        --rounds 1 --salt-seed 9502 --tag wu_pf$L >/dev/null 2>&1
      echo "warmup: ③ prefill ${L} 完成（JIT 特化预编译）"
    done
  fi
F1=$(timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1); F1=${F1:-n/a}
echo "warmup: done ✓ free=${F1} MiB"
exit 0
