#!/bin/bash
# prof_guard.sh [seconds] [unit] — bounded torch-profiler session for sm70-serve.
#
# Why: the daemon's steady-state RSS is 121-125 GB of 123 GB physical (todo 277),
# so a profiler that buffers events in-process (kineto) has no headroom. On
# 2026-10-22 a with_stack collection over a 4K prefill grew the worker to
# 12.2 GB anon-rss and the box OOMed, killing VLLM::Worker_TP.
#
# Rules enforced here (not "documented again" — enforced):
#   1. refuse to profile when available RAM < MIN_AVAIL_GB (default 20);
#   2. refuse with_stack (per-event Python stacks; unbounded growth);
#   3. always auto-stop after SECONDS (default 20), even if the caller dies.
set -u
SECS="${1:-20}"
UNIT="${2:-sm70-serve}"
MIN_AVAIL_GB="${MIN_AVAIL_GB:-20}"
API="http://localhost:8001"
KEY="${SM70_API_KEY:-}"

avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "${avail:-0}" -lt "$MIN_AVAIL_GB" ]; then
  echo "REFUSE: available ${avail}G < ${MIN_AVAIL_GB}G (daemon baseline is 121-125G of 123G)"
  exit 3
fi
if grep -q '"torch_profiler_with_stack": true' /root/run/run_sm70_serve_ipc.sh; then
  echo "REFUSE: with_stack=true in the serve script; unbounded per-event stacks OOM this box"
  exit 4
fi

start=$(date +%s)
# Bound the session by ITERATIONS, not by wall time: the profiler buffers every
# step it sees, and a 200 s session (a live 8192-token client once blocked the
# stop for 190 s) inflated each worker's anon-RSS by ~10 GB. max_iterations makes
# the profiler stop itself after N steps, so the buffer stays small regardless
# of what the client does.
ITERS="${ITERS:-${3:-20}}"
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"max_iterations\": ${ITERS}}" "$API/start_profile" \
  -o /dev/null -w "start=%{http_code} iters=%{size_download}\n" || exit 5
echo "bounded profile: max_iterations=$ITERS (the kernel stops itself); wall timer ${SECS}s is only a safety net"
# Hard stop: run the timer in a child so `set -u` scripts cannot leak a session.
( sleep "$SECS"; curl -s -X POST -H "Authorization: Bearer $KEY" "$API/stop_profile" -o /dev/null -w "auto-stop=%{http_code}\n" ) &
TIMER=$!
echo "profiling for ${SECS}s (avail ${avail}G); caller should drive requests now"
wait $TIMER
echo "elapsed $(( $(date +%s) - start ))s; avail now $(free -g | awk '/^Mem:/{print $7}')G"
