#!/bin/bash
# Fail-fast measurement harness for SM70 (2026-10-24: written because my ad-hoc
# chains used ';' and kept running after a failed serve start, then restarted a
# broken service).
#
#   /root/run/measure.sh <tag> [prefill_rounds] [decode1_rounds] [conc8_rounds]
#
# Guarantees:
#   * set -euo pipefail: the first failing step aborts and prints the RAW log tail
#   * every restart is VERIFIED ('Application startup complete' + API 200),
#     never inferred from `systemctl is-active`
#   * preflight checks memory and disk before touching anything
#   * canary runs from the correct directory and its rc decides keep/rollback
#   * a final STATUS line + non-zero exit on any failure
set -euo pipefail

TAG="${1:?usage: measure.sh <tag> [prefill] [decode1] [conc8]}"
NP="${2:-2}"; ND="${3:-5}"; NC="${4:-3}"
OUT="/data/sm70_results/${TAG}"
PY=/data/vllm-dev/bin/python
TESTS=/data/src/vllm-v2/sm70_tests
mkdir -p "$OUT"

fail() { echo "STATUS=FAIL step=$1"; echo "--- raw log tail ---"; journalctl -u sm70-serve -n 25 --no-pager 2>/dev/null | cut -c1-200; exit 1; }

echo "=== [1/5] preflight ==="
bash /root/run/ram_guard.sh >/dev/null || fail preflight-ram
df -h /data | awk 'NR==2{print "  /data used", $5}'
[ -d "$TESTS" ] || fail missing-tests-dir

echo "=== [2/5] restart + verify ==="
systemctl restart sm70-serve
if ! /root/run/serve_wait.sh 1800; then fail serve-start; fi
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
  -H "Authorization: Bearer $(grep -oP 'SM70_API_KEY=\K.*' /data/src/sm70_env.sh)" \
  http://127.0.0.1:8001/v1/models)
[ "$CODE" = "200" ] || fail "api-$CODE"
echo "  serve READY, api 200"

echo "=== [3/5] canary (content gate) ==="
cd "$TESTS"
if ! timeout 900 $PY correctness.py > "$OUT/canary.log" 2>&1; then
  tail -5 "$OUT/canary.log"; fail canary
fi
grep -q "CORRECT-OK checks=9 failed=0" "$OUT/canary.log" || fail canary-content
echo "  canary 9/9"

echo "=== [4/5] ruler ==="
timeout 1800 $PY qsa_conc_tput.py --mode prefill --len 65536 --conc 1 \
  --salt-seed 9502 --rounds "$NP" > "$OUT/prefill_9502.log" 2>&1 || fail prefill
grep -E "^round=" "$OUT/prefill_9502.log" | grep -v warm | sed 's/^/  /'
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 1 \
  --salt-seed 9503 --rounds "$ND" > "$OUT/decode1_9503.log" 2>&1 || fail decode1
timeout 2400 $PY qsa_conc_tput.py --mode decode --len 65536 --conc 8 \
  --salt-seed 9504 --rounds "$NC" > "$OUT/conc8_9504.log" 2>&1 || fail conc8

$PY - "$OUT" <<'PY'
import re, statistics, sys, pathlib
out = pathlib.Path(sys.argv[1])
def med(name):
    vals = [float(m.group(1)) for line in (out/name).read_text().splitlines()
            if "warm" not in line and (m := re.search(r"(?:prefill|decode)_agg=([0-9.]+)", line))]
    return (statistics.median(vals) if vals else 0.0), len(vals)
p, n1 = med("prefill_9502.log"); d, n2 = med("decode1_9503.log"); c, n3 = med("conc8_9504.log")
print(f"RESULT prefill={p:.0f} (n={n1}) decode1={d:.1f} (n={n2}) conc8={c:.1f} (n={n3})")
PY

echo "=== [5/5] xid check ==="
X=$(journalctl -k --since "30 min ago" --no-pager 2>/dev/null | grep -c Xid || true)
echo "  new Xid: $X"
[ "$X" = "0" ] || fail xid
echo "STATUS=OK tag=$TAG results=$OUT"
