#!/bin/bash
# Standard decode concurrency metric: conc=8 and conc=16, same process era,
# distinct salts (never 9502, which is the prefill ruler).
#   usage: conc_decode.sh <tag> <salt-base>
# Reads whatever config the service is currently running; call once per arm
# after serve_wait_q36.sh reports READY.
set -u
TAG=${1:?tag}; SB=${2:?salt base}
export SM70_BASE=http://127.0.0.1:8001
export SM70_MODEL_PATH=/data/models/Qwen3.6-35B-A3B-NVFP4-Fast
export SM70_MODEL=Qwen3.6-35B-A3B
export SM70_API_KEY="$(cat /root/.vllm_api_key)"
cd /data/src/vllm-v2/sm70_tests
for C in 8 16; do
  S=$((SB + C))
  printf '[%s] conc=%d start %s\n' "$TAG" "$C" "$(date +%H:%M:%S)"
  timeout 2400 /data/vllm-dev/bin/python qsa_conc_tput.py --mode decode \
    --conc "$C" --salt-seed "$S" --rounds 3 --tag "${TAG}_c${C}" 2>&1 |
    grep -E "^round=" | sed "s/^/  [$TAG c$C] /"
done
echo "DONE_CONC_$TAG"
