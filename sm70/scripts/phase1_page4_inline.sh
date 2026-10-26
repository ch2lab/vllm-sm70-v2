#!/usr/bin/env bash
# Phase-1 experiment 2: page4 block-table sort fused into the table kernel
# (tl.sort, SORT_INLINE) — canonical order (deterministic) without the separate
# torch.sort launch (12 radixSort kernels/step, 61.7 us each).
set -uo pipefail
source /data/src/sm70_env.sh
OUT=/data/sm70_results/phase1-page4-inline
mkdir -p "$OUT"
cd /root/run
cp -a run_sm70_serve_sync_nomtp.sh "$OUT/script.bak"

/data/vllm-dev/bin/python - <<'PY'
import pathlib
p = pathlib.Path('/root/run/run_sm70_serve_sync_nomtp.sh')
s = p.read_text()
assert s.count('export VLLM_SM70_QSA_PAGE4_SORT=skip') == 1
p.write_text(s.replace('export VLLM_SM70_QSA_PAGE4_SORT=skip',
                       'export VLLM_SM70_QSA_PAGE4_SORT=inline'))
print('mode -> inline')
PY

systemctl stop sm70-serve && sleep 2
/root/run/ram_guard.sh 30 page4inline || exit 3
systemctl start sm70-serve
timeout 1500 bash /root/run/serve_wait.sh 1400 sm70-serve || exit 1

echo "=== decode1 anchor (salt 9503, 6 rounds) ==="
cd /data/src/vllm-v2/sm70_tests
PY=/data/vllm-dev/bin/python
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds 6 > "$OUT/decode1_9503.log" 2>&1
grep -E "^round=" "$OUT/decode1_9503.log" | grep -v warm | sed 's/^/  /'
$PY - "$OUT/decode1_9503.log" <<'PY'
import re, sys, statistics
vals = [float(m.group(1)) for line in open(sys.argv[1])
        if (m := re.search(r"decode_agg=([0-9.]+)", line)) and "warm" not in line]
print(f"  decode1 INLINE: n={len(vals)} median={statistics.median(vals):.1f} "
      f"min={min(vals):.1f} vals={[f'{v:.1f}' for v in vals]}")
PY

echo "=== content gates ==="
SM70_RESULTS_DIR="$OUT" timeout 1200 $PY correctness.py 2>&1 | tail -12 | sed 's/^/  /'
timeout 900 $PY long_content_gate.py 2>&1 | tail -4 | sed 's/^/  /'
dmesg 2>/dev/null | grep -ci Xid | sed 's/^/  dmesg Xid lines: /'
