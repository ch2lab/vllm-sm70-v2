#!/usr/bin/env bash
# SM70 (V100) e2e serve — SYNCED tree (/data/src/vllm-v2, upstream main merged),
# 8x V100, port 8001, TP8. Interpreter /data/vllm-dev (system-site-packages +
# humming-kernels 0.1.17); PYTHONPATH puts the synced tree ahead of the editable
# install. Everything else is byte-identical to run_sm70_serve_ipc.sh.
#
# Reference: the 1Cat production unit (sm70_docs/archive-1cat-20261012/vllm.service),
# which is already 8001 + TP8 + nvfp4 KV + nvfp4 PLE. Differences vs that unit:
#   * interpreter = /data/vllm-dev (editable vllm -> /data/src/vllm-v2; flashinfer fork
#     -> /data/src/flashinfer), NOT /data/vllm-1cat.
#   * The 1Cat campaign switches (QSA/XQA/AR/TM/TOPK2/HC/ROWMAJOR/XQA_MIN_BLOCKS) are
#     absent from the clean tree's envs.py -> not set here (they would be no-ops).
#   * nvfp4 PLE is config-driven in the new tree (text_config.ple_embedding_dtype="nvfp4"
#     in the model config.json), not an env; VLLM_PLE_EMBEDDING_DTYPE is unused here.
#   * FLASHINFER_DISABLE_VERSION_CHECK=1: flashinfer-python 0.7.0 vs flashinfer-cubin
#     0.6.18 skew (pre-existing packaging gap; the SM70 JIT path doesn't use the cubin
#     gate, so the version check is bypassed rather than patched).
#
# Prerequisite: the SM70 flashinfer.py backend merge (Builder + Impl) must be complete
# and the _C extension built (build_sm70.sh) before this will serve SM70 attention.
set -euo pipefail

# CUDA 12.8 toolkit + LD_LIBRARY_PATH (load-bearing) + TORCH_CUDA_ARCH_LIST.
source /data/src/sm70_env.sh

export FLASHINFER_DISABLE_VERSION_CHECK=1
# Long boot: TP8 + 262k context + kernel JIT warmup.
export VLLM_ENGINE_READY_TIMEOUT_S=1400

# Run outside /data/src so the editable install isn't shadowed by a sibling vllm/ dir
# (the frozen /data/src/vllm tree would otherwise be picked up as a namespace package).
cd /data

# The 12 QSA layers (qwen_sparse_attention, idx 3,7,...,47) keep their own
# fp16 main KV: the SM70 QSA Triton kernel reads BF16/FP16/FP8-e4m3 caches,
# not the nvfp4 packed layout (the nvfp4-KV-QSA kernel lived only in the
# excluded 1Cat lineage). Full nvfp4 KV: one --kv-cache-dtype nvfp4.
# NOTE: no comment lines inside the exec continuation chain below — a
# comment mid-chain would truncate the command at the '#' (bash).
touch /tmp/page4_tp8_mode   # C-2 M2 WIN 10-15: page4-TP8 route (valve, survives restarts)
touch /tmp/page4_tp8_mode && echo local > /tmp/qsa_page4_backend   # P0 10-16: in-tree stable-ABI page4 backend (valve, survives restarts)
# 10-17 定案: 生产性能阀 (与 sm70-serve.service drop-in 同值) 固化进脚本 —
# nohup/手动启动不得再绕过 unit 层丢阀 (13% 缺口事故根因, EXP12F_PROGRESS.md).
export SM70_QSA_FMA=1
export VLLM_SM70_AR_HIER=1
export VLLM_SM70_AR_HIER_FILE=/root/run/sm70/ar_hier_mode
# TEMP-AB 2026-10-21 (cap-16k window): hijack-size census, log-only. REMOVE.
export VLLM_SM70_AR_HIER_TRACE=1
export VLLM_SM70_AR_WIRE=3
export VLLM_SM70_AR_ADAPTIVE=1
export VLLM_SM70_TRITON_TOPK=1
export VLLM_SM70_QWEN38_FP16_GEMV=1
export VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=0
export VLLM_SM70_QWEN38_FUSED_HC_UP_GATE=1
export VLLM_SM70_QWEN38_FUSED_HC_DOWN_SILU=1
export VLLM_SM70_QSA_VERIFY_PAGE_CAP=512

export VLLM_SM70_MOE_QPN2_NS=1
# GDN K2 chunked shadow kernel valve. DEFAULT OFF (0).
# 2026-10-17: attempted adoption -> IMA (Xid 31 MMU VIRT_WRITE, all 8 GPUs)
# at engine init under serving shapes; the offline G2 gate (T=8192,
# initial_state=None) missed it. DO NOT enable until the OOB write is fixed.
export VLLM_SM70_GDN_CHUNKED=1
# EXP-12 10-16 晚: XQA vbt 观测审计 (纯观测, 定案后回滚此 env)
# EXP-12b 10-16 晚: 同步 launch — IMA 在故障 kernel 的 launch 处同步暴露,
# journal 中故障 rank 的最后一条 "XQA vbt audit ok" 行即故障调用 (定案后回滚)
# P0 FIX (10-17): PLE UVA lookup side-stream race — ngram_embedding.start_prefetch
# puts the lookup on a side stream writing the single persistent _prefetch_buffer
# while step k's _finalize_prefetch still reads it. VLLM_SM70_PLE_SYNC=1 serializes
# the lookup onto the current stream (production-intended value, 01_deployment.md).
# This export was dead code at line 87 (after exec); moved before exec to take effect.
export VLLM_SM70_PLE_SYNC=1
# 10-18 裁决 (user): prefill/decode 速度一律取服务端响应体数据 (usage 计数 + metrics 计时),
# 不再用客户端本地统计。--enable-per-request-metrics 使响应体 / 流式末块携带
# PerRequestMetrics (time_to_first_token_ms / generation_time_ms / mean_itl_ms /
# tokens_per_second); 纯报数, 不改调度/模型/性能; 与 --disable-log-stats 互斥 (本服务未用)。
export VLLM_SM70_QSA_POS_FUSE=1
export VLLM_SM70_QSA_STAGES=3
export VLLM_SM70_QSA_DECODE_FUSE=0
export VLLM_SM70_QSA_PLAN_PACK=1
export VLLM_SM70_QSA_PAGE4_SORT=inline32
export VLLM_SM70_MOE_QPN2_WIDE=256
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128   # tiling knob: shrink the prefill logits workspace 512->128 MiB
export VLLM_SM70_ALLOW_DUP_LAYER=1
export PYTHONPATH=/data/src/vllm-v2
exec /data/vllm-dev/bin/python -m vllm.entrypoints.cli.main serve \
  /data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4 \
  --load-format ipc_cache \
  --served-model-name Qwen3.8-Flash-Next \
  --trust-remote-code \
  --dtype float16 \
  --kv-cache-dtype nvfp4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 64 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 \
  --gpu-memory-utilization 0.87 \
  --kv-cache-memory-bytes 9600000000 \
  --enable-prefix-caching \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/root/run/prof", "torch_profiler_with_stack": false, "torch_profiler_record_shapes": false}' \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-per-request-metrics \
  --port 8001 \
  --disable-uvicorn-access-log \
  --allowed-origins '["*"]' \
  --api-key "${SM70_API_KEY:?SM70_API_KEY not set - export it in /data/src/sm70_env.sh}"
export VLLM_SM70_PLE_SYNC=1        # PLE prefetch side-stream race documented in tree; serializes lookup onto current stream
