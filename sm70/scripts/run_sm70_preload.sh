#!/usr/bin/env bash
# SM70 weight-cache daemon (Arm C) — one preload daemon per GPU (TP8).
# Model/parallelism args MUST mirror run_sm70_serve_ipc.sh: the IPC protocol
# fingerprint-checks the cache config between daemon and engine.
# PLE via pre-quantized sidecar (pure-CPU read; NVFP4 codes/scales over IPC).
set -euo pipefail

# CUDA 12.8 toolkit + LD_LIBRARY_PATH (load-bearing) + TORCH_CUDA_ARCH_LIST.
source /data/src/sm70_env.sh

export FLASHINFER_DISABLE_VERSION_CHECK=1
# Long boot: TP8 + 262k context + kernel JIT warmup.
export VLLM_ENGINE_READY_TIMEOUT_S=1400

# Run outside /data/src so the editable install isn't shadowed by a sibling vllm/ dir.
cd /data

# NOTE: `vllm preload` takes the model via --model (positional form is serve-only).
export VLLM_SM70_ALLOW_DUP_LAYER=1

# Fail closed if flashinfer is not our local SM70 build (PyPI 0.7.2 lacks it).
/data/vllm-dev/bin/python - <<'EOF' || { echo "FATAL: flashinfer is not /data/src/flashinfer (SM70 backend) -- refusing to start"; exit 1; }
import os, sys, flashinfer
sys.exit(0 if os.path.dirname(flashinfer.__file__).startswith('/data/src/flashinfer') else 1)
EOF
exec /data/vllm-dev/bin/python -m vllm.entrypoints.cli.main preload \
  --model /data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4 \
  --trust-remote-code \
  --dtype float16 \
  --kv-cache-dtype nvfp4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-batched-tokens 8 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.87 \
  --kv-cache-memory-bytes 5200000000 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --no-enable-flashinfer-autotune \
  --weight-cache-health-port 8002
