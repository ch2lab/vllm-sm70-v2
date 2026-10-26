#!/bin/bash
# MoE wide occupancy matrix (2026-10-24): env-only knobs, one prefill round per
# config. NS/K4 do not change the weight prepack layout (cheap); WIDE does.
set -uo pipefail
LOG=/data/sm70_results/moe_wide_matrix.log
exec >>"$LOG" 2>&1
echo "=== $(date -Is) MoE wide matrix start ==="
SVC=/root/run/run_sm70_serve_ipc.sh
PY=/data/vllm-dev/bin/python
TESTS=/data/src/vllm-v2/sm70_tests
cp -a "$SVC" /tmp/opencode/serve_pre_moe_matrix.sh

run_cfg() {  # $1=tag $2=NS $3=K4
  local tag="$1" ns="$2" k4="$3"
  echo "--- config $tag (NS=$ns K4=$k4) ---"
  sed -i '/VLLM_SM70_MOE_QPN2_/d' "$SVC"
  printf 'export VLLM_SM70_MOE_QPN2_NS=%s\nexport VLLM_SM70_MOE_QPN2_K4=%s\n' "$ns" "$k4" >> "$SVC"
  systemctl restart sm70-serve
  if ! /root/run/serve_wait.sh 1800; then
    echo "  $tag SERVE_FAILED"; journalctl -u sm70-serve -n 12 --no-pager | cut -c1-160
    return 1
  fi
  cd "$TESTS"
  timeout 1200 $PY qsa_conc_tput.py --mode prefill --len 65536 --conc 1 \
    --salt-seed 9502 --rounds 2 2>&1 | grep -E "^round=" | grep -v warm | sed "s/^/  $tag /"
  grep -oE "MoE shadow path active: [^\"]*" /data/sm70_results/tp4_serve.log 2>/dev/null | tail -1
  journalctl -u sm70-serve --since "5 min ago" --no-pager 2>/dev/null | grep -oE "MoE shadow path active:.*" | tail -1
}

run_cfg base 0 0
run_cfg ns1  1 0
run_cfg k4   0 1
run_cfg ns1k4 1 1

echo "--- restore (drop the env, restart, canary) ---"
cp -a /tmp/opencode/serve_pre_moe_matrix.sh "$SVC"
systemctl restart sm70-serve
/root/run/serve_wait.sh 1800 && echo "  restored" || echo "  RESTORE FAILED"
cd "$TESTS" && timeout 900 $PY correctness.py 2>&1 | tail -2
echo "=== $(date -Is) MoE wide matrix done ==="
