#!/bin/bash
# TP4 vs TP8 A/B (2026-10-24 user request): one TP4 replica on GPUs 0-3, same model,
# same ruler. The daemon holds the TP8 ipc cache in VRAM on all 8 GPUs, so it must
# be stopped first; TP4 then loads from disk (slow boot, no ipc cache).
set -uo pipefail
LOG=/data/sm70_results/tp4_ab4.log
exec >>"$LOG" 2>&1
echo "=== $(date -Is) TP4 A/B start ==="

PY=/data/vllm-dev/bin/python
TESTS=/data/src/vllm-v2/sm70_tests
MODEL=/data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4
KEY=$(grep -oP 'SM70_API_KEY=\K.*' /data/src/sm70_env.sh)
source /data/src/sm70_env.sh

echo "--- stop serve + daemon (free VRAM) ---"
systemctl stop sm70-serve
systemctl stop sm70-daemon
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4

echo "--- launch TP4 on GPU 0-3 (NCCL AR, no custom AR: world_size=4) ---"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SM70_AR_HIER=0
export VLLM_SM70_QSA_STAGES=3
export VLLM_SM70_QSA_FLASHV100=1   # 2026-10-24: TP4 page4 route (heads==6 gate)
export SM70_QSA_FMA=1
export VLLM_SM70_QSA_PAGE4_SORT=inline32
export VLLM_SM70_QWEN38_FP16_GEMV=1
export VLLM_SM70_QWEN38_FUSED_HC_UP_GATE=1
export VLLM_SM70_QWEN38_FUSED_HC_DOWN_SILU=1
export VLLM_SM70_QSA_VERIFY_PAGE_CAP=512
export VLLM_SM70_QSA_POS_FUSE=1
export VLLM_SM70_QSA_PLAN_PACK=1
export VLLM_SM70_QSA_DRAFT_META_UPDATE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_READY_TIMEOUT_S=2400

nohup $PY -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name Qwen3.8-Flash-Next \
  --trust-remote-code --dtype float16 --kv-cache-dtype nvfp4 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --enable-hc-sp --max-model-len 262144 --max-num-batched-tokens 10240 \
  --max-num-seqs 32 --cudagraph-capture-sizes 1 2 4 8 16 32 \
  --gpu-memory-utilization 0.95 --kv-cache-memory-bytes 4000000000 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":true}' \
  --no-enable-flashinfer-autotune --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-per-request-metrics --port 8001 --disable-uvicorn-access-log \
  --allowed-origins '["*"]' --api-key "$KEY" > /data/sm70_results/tp4_serve.log 2>&1 &
SPID=$!
echo "  tp4 pid=$SPID"

for i in $(seq 1 180); do
  if grep -q "Application startup complete" /data/sm70_results/tp4_serve.log 2>/dev/null; then
    echo "  TP4 READY after $((i*10))s"; break
  fi
  if ! kill -0 $SPID 2>/dev/null; then echo "  TP4 DIED"; tail -20 /data/sm70_results/tp4_serve.log; exit 1; fi
  sleep 10
done
grep -oE "Prepared [0-9]+ Qwen3.8 checkpoint-FP16 SM70 M=1 GEMV projections" /data/sm70_results/tp4_serve.log | tail -1
grep -q "Application startup complete" /data/sm70_results/tp4_serve.log || { echo "  TP4 NOT READY"; tail -20 /data/sm70_results/tp4_serve.log; kill $SPID; exit 1; }

cd "$TESTS"
echo "--- TP4 ruler ---"
timeout 1800 $PY qsa_conc_tput.py --mode prefill --len 65536 --conc 1 --salt-seed 9502 --rounds 2 2>&1 | grep -E "^round=" | grep -v warm
timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 --salt-seed 9503 --rounds 4 2>&1 | grep -E "^round=" | grep -v warm
timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 8 --salt-seed 9504 --rounds 3 2>&1 | grep -E "^round=" | grep -v warm

echo "--- stop TP4, restore daemon + TP8 serve ---"
kill $SPID 2>/dev/null; sleep 10
kill -9 $SPID 2>/dev/null; sleep 5
systemctl start sm70-daemon
for i in $(seq 1 90); do
  journalctl -u sm70-daemon --since "20 min ago" --no-pager 2>/dev/null | grep -q "Weight cache daemon READY" && break
  sleep 20
done
echo "  daemon: $(systemctl is-active sm70-daemon)"
rm -rf /root/.cache/torch_extensions/py312_cu128/*
systemctl start sm70-serve
/root/run/serve_wait.sh 2400 && echo "  TP8 restored" || echo "  TP8 restore FAILED"
cd "$TESTS" && timeout 900 $PY correctness.py 2>&1 | tail -2
echo "=== $(date -Is) TP4 A/B done ==="
