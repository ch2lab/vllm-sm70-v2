#!/usr/bin/env bash
# W15: steady-state conc8 measurement with a FIXED salt (same prompt => APC hit),
# warm runs discarded, then median + range of the measured runs.
# usage: conc8_steady.sh <salt> [n_warm=3] [n_meas=3] [groups=3]
set -uo pipefail
SALT="${1:?salt}"
NW="${2:-3}"; NM="${3:-3}"; NG="${4:-3}"
cd /data/src/vllm-v2/sm70_tests || exit 1
source /data/src/sm70_env.sh >/dev/null 2>&1
PY=/data/vllm-dev/bin/python
one() { $PY qsa_conc_tput.py --mode decode --conc 8 --len 65536 --salt-seed "$SALT" --rounds 2 2>/dev/null \
        | sed -n 's/.*round=1 .*decode_agg=\([0-9.]*\).*/\1/p' | head -1; }
for g in $(seq 1 "$NG"); do
  for i in $(seq 1 "$NW"); do one >/dev/null; done
  vals=""
  for i in $(seq 1 "$NM"); do v=$(one); vals="$vals $v"; echo "  group=$g meas=$i agg=$v"; done
  $PY - "$vals" "$g" <<'PY'
import statistics, sys
v = [float(x) for x in sys.argv[1].split()]
print(f"  group={sys.argv[2]} median={statistics.median(v):.1f} range={min(v):.1f}-{max(v):.1f} "
      f"spread={100*(max(v)-min(v))/statistics.median(v):.1f}%")
PY
done
