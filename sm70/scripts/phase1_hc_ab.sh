#!/usr/bin/env bash
# Paired A/B for the HC fusion fix: same tree, only the two valves flip.
# A: fusions OFF (env=0)  B: fusions ON (env=1). Cross-boot drift ±2-3% makes a
# single-arm comparison useless, so the pair is measured back to back.
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/phase1-hc-ab
mkdir -p "$OUT"
PY=/data/vllm-dev/bin/python
cd /data/src/vllm-v2/sm70_tests

set_valves() {  # $1 = 0|1
  $PY - "$1" <<'PY'
import pathlib, sys
v = sys.argv[1]
p = pathlib.Path('/root/run/run_sm70_serve_sync_nomtp.sh')
s = p.read_text()
for name in ("VLLM_SM70_QWEN38_FUSED_HC_UP_GATE", "VLLM_SM70_QWEN38_FUSED_HC_DOWN_SILU"):
    old = f"export {name}={'1' if v == '0' else '0'}"
    new = f"export {name}={v}"
    assert s.count(old) == 1, (name, s.count(old))
    s = s.replace(old, new)
p.write_text(s)
print(f"valves -> {v}")
PY
}

measure() {  # $1 = tag
  timeout 1800 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
    --salt-seed 9503 --rounds 5 > "$OUT/decode1_$1.log" 2>&1
  $PY - "$OUT/decode1_$1.log" "$1" <<'PY'
import re, sys, statistics
vals = [float(m.group(1)) for line in open(sys.argv[1])
        if (m := re.search(r"decode_agg=([0-9.]+)", line)) and "warm" not in line]
print(f"  {sys.argv[2]}: n={len(vals)} median={statistics.median(vals):.1f} "
      f"vals={[f'{v:.1f}' for v in vals]}")
PY
}

boot() {
  systemctl stop sm70-serve && sleep 2
  /root/run/ram_guard.sh 30 "$1" || exit 3
  systemctl start sm70-serve
  timeout 1500 bash /root/run/serve_wait.sh 1400 sm70-serve || exit 1
}

for arm in off on; do
  if [ "$arm" = off ]; then set_valves 0; else set_valves 1; fi
  boot "hcab_$arm"
  echo "=== arm $arm ==="
  measure "$arm"
  journalctl -u sm70-serve --since "8 min ago" --no-pager \
    | grep -aE "fused FP16 HC (up\+gate|down\+SiLU)|no targets" | cut -c60-150 | tail -2 | sed 's/^/  /'
done
