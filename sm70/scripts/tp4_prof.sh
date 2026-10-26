#!/bin/bash
# TP4 profiler run (2026-10-24): find the kernel that scales worse than 2x.
# TP8 prefill is 7.54 s (8687 tok/s); perfect 2x scaling would be 15.1 s (4343);
# TP4 measured 19.9 s (3297) = 2.64x, so ~4.8 s is super-linear somewhere.
set -uo pipefail
LOG=/data/sm70_results/tp4_prof16.log
exec >>"$LOG" 2>&1
echo "=== $(date -Is) TP4 profiler run start ==="
PY=/data/vllm-dev/bin/python
TESTS=/data/src/vllm-v2/sm70_tests
MODEL=/data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4
KEY=$(grep -oP 'SM70_API_KEY=\K.*' /data/src/sm70_env.sh)
source /data/src/sm70_env.sh

systemctl stop sm70-serve
systemctl stop sm70-daemon
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SM70_AR_HIER=0
export VLLM_SM70_QSA_STAGES=3
export VLLM_SM70_QSA_FLASHV100=1   # TP4 6-head page4 gate
echo local > /tmp/qsa_page4_backend  # arm the in-tree stable-ABI page4 ops (only source)
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
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/run/prof","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_activities":["CUDA"]}' \
  --no-enable-flashinfer-autotune --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-per-request-metrics --port 8001 --disable-uvicorn-access-log \
  --allowed-origins '["*"]' --api-key "$KEY" > /data/sm70_results/tp4_prof_serve.log 2>&1 &
SPID=$!
echo "  tp4 pid=$SPID"

for i in $(seq 1 180); do
  grep -q "Application startup complete" /data/sm70_results/tp4_prof_serve.log 2>/dev/null && { echo "  READY after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "  DIED"; tail -20 /data/sm70_results/tp4_prof_serve.log; exit 1; }
  sleep 10
done
grep -q "Application startup complete" /data/sm70_results/tp4_prof_serve.log || { echo "  NOT READY"; kill $SPID; exit 1; }

cd "$TESTS"
echo "--- profiled prefill ---"
curl -s -o /dev/null -X POST -H "Authorization: Bearer $KEY" http://127.0.0.1:8001/start_profile -w "  start=%{http_code}\n" --max-time 60
timeout 1200 $PY qsa_conc_tput.py --mode prefill --len 65536 --conc 1 --salt-seed 9502 --rounds 1 2>&1 | grep -E "^round=" | sed 's/^/  /'
curl -s -o /dev/null -X POST -H "Authorization: Bearer $KEY" http://127.0.0.1:8001/stop_profile -w "  stop=%{http_code}\n" --max-time 900

echo "--- per-kernel (TP4) ---"
T=$(ls -t /root/run/prof/*rank0*.gz | head -1)
$PY - "$T" <<'EOF' 2>&1 | head -16
import gzip, json, sys, statistics, collections
with gzip.open(sys.argv[1],"rt") as f: d=json.load(f)
agg=collections.defaultdict(lambda:[0,0,[]])
for e in d["traceEvents"]:
    if e.get("cat")!="kernel": continue
    a=agg[e["name"]]; a[0]+=e.get("dur",0); a[1]+=1; a[2].append(e.get("dur",0))
print(f"{'total_ms':>9} {'calls':>7} {'med_us':>8}  kernel")
for n,(t,c,v) in sorted(agg.items(), key=lambda kv:-kv[1][0])[:12]:
    print(f"{t/1e3:9.1f} {c:7d} {statistics.median(v):8.2f}  {n[:56]}")
EOF

echo "--- correctness gate on TP4 (same boot) ---"
overall=0
if timeout 1800 $PY correctness.py 2>&1 | tee /data/sm70_results/tp4_gate_canary.log | tail -3; then
  grep -q "CORRECT-OK checks=9 failed=0" /data/sm70_results/tp4_gate_canary.log || overall=1
else
  overall=1
fi
echo "  canary on TP4: $([ $overall -eq 0 ] && echo PASS || echo FAIL)"

echo "--- stop TP4 (bench left idle; TP8 baseline starts only when asked) ---"
kill $SPID 2>/dev/null; sleep 10; kill -9 $SPID 2>/dev/null; sleep 3
echo "=== $(date -Is) done rc=$overall ==="
exit $overall
