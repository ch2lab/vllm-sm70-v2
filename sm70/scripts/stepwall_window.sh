#!/bin/bash
# decode step-wall window (P5): nsys on serve with FULL production env
# (lesson G0 v4 #11886: window env must match prod drop-in or data is an artifact).
# Safe-fallback: any failure path ends with the service back up.
#
# PROTOCOL (2026-10-14 incident fix):
#  - The window kills ONLY its own process tree (setsid session -> kill -- -PGID).
#  - Long-lived components (sm70-daemon) are READ-ONLY here: pids recorded
#    up front, liveness re-checked at the end. Any dead daemon pid = protocol
#    violation -> FATAL, never a silent "clean".
#  - No bare pattern kills. The daemon's cmdline shares the
#    "vllm.entrypoints.cli.main" prefix with serve (subcommands serve/preload
#    differ) — a prefix pattern kill double-kills (root cause 2026-10-14).
OUT=/data/sm70_results/round5/p5_decodewall
mkdir -p $OUT
cd $OUT

# --- long-lived component bookkeeping (read-only baseline) -----------------
DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
echo "daemon pids (read-only baseline): ${DAEMON_PIDS:-NONE}"
[ -n "$DAEMON_PIDS" ] || { echo "FATAL: no daemon running before window — refuse to start"; exit 1; }

cleanup() {
  # never leave the service down (unit ordering now enforces the daemon gate:
  # sm70-serve Requires+After sm70-daemon, daemon active = 8002 ready)
  if ! systemctl is-active -q sm70-serve; then
    echo "cleanup: service not active -> restarting"
    systemctl start sm70-serve
  fi
}
trap cleanup EXIT

echo "=== window start $(date) ==="
echo "env: SM70_QSA_FMA=1 VLLM_PLE_QUANT_DIR set (matches unit + drop-in)"
systemctl stop sm70-serve
sleep 5

export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
export SM70_QSA_FMA=1
export VLLM_PLE_QUANT_DIR=/data/models/Qwen3.8-Flash-Next-NVFP4/ple-sidecar/ples_nvfp4

echo "=== nsys boot $(date) ==="
PGID_FILE=$OUT/nsys.pgid
rm -f "$PGID_FILE"
# setsid: whole nsys tree in its own session; leader pid (== pgid) recorded
# by the wrapper BEFORE exec so we can kill exactly this tree later.
setsid bash -c "echo \$\$ > $PGID_FILE; exec nsys profile -o $OUT/stepwall --force-overwrite=true /root/run/run_sm70_serve_ipc.sh" \
  > $OUT/boot.log 2>&1 &
for i in $(seq 1 20); do [ -s "$PGID_FILE" ] && break; sleep 0.5; done
NSYS_PID=$(cat "$PGID_FILE" 2>/dev/null)
echo "nsys session leader (pgid) = $NSYS_PID"
[ -n "$NSYS_PID" ] || { echo "FATAL: nsys tree not tracked"; exit 1; }

READY=0
for i in $(seq 1 150); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "READY=1 after ~$((i*10))s"
    READY=1
    break
  fi
  sleep 10
done
[ $READY -eq 1 ] || { echo "FATAL: not ready in 1500s"; kill -9 -- -"$NSYS_PID" 2>/dev/null; exit 1; }

echo "=== ruler 65K (decode-focused) $(date) ==="
cd /data/src/vllm-v2
timeout 900 /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 65536 > $OUT/ruler.log 2>&1
RULER_RC=$?
echo "RULER_RC=$RULER_RC"
cat $OUT/ruler.log

echo "=== nsys stop $(date) ==="
# SIGINT the GROUP: nsys finalizes the rep; serve tree (its children) stays up
kill -INT -- -"$NSYS_PID" 2>/dev/null
# bounded finalize wait (G0 v4: finalize can hang; rep written incrementally)
for i in $(seq 1 15); do
  kill -0 -- -"$NSYS_PID" 2>/dev/null || break
  sleep 10
done
kill -9 -- -"$NSYS_PID" 2>/dev/null   # rest of the window tree (serve + workers)
sleep 5
# stray-serve check: SERVE-subcommand pattern only (cannot match the daemon's
# "preload" cmdline). Any survivor is a window bug, not something to sweep.
SRV_LEFT=$(pgrep -f "vllm.entrypoints.cli.main serve" || true)
[ -n "$SRV_LEFT" ] && echo "FATAL: stray serve pids: $SRV_LEFT" || echo "no stray serve processes"
# daemon liveness (read-only): every baseline pid must still be alive
DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: PROTOCOL VIOLATION — daemon pids dead:$DAEMON_DEAD" || echo "daemon liveness OK (untouched)"
echo "nsys exited"

ls -la $OUT/stepwall.nsys-rep 2>/dev/null
timeout 300 nsys stats --report cuda_gpu_kern_sum $OUT/stepwall.nsys-rep > $OUT/kern_sum.txt 2>/dev/null \
  && echo "STATS OK" || echo "STATS_FAILED (rep may be incomplete; rescue later)"
timeout 300 nsys stats --report cuda_api_sum $OUT/stepwall.nsys-rep > $OUT/api_sum.txt 2>/dev/null \
  && echo "API STATS OK" || echo "API_STATS_FAILED"
# sqlite for timestamp segmentation (prefill vs decode) — sqlite tables
# carry per-kernel start/end; GDN prof used the same path
timeout 600 nsys export --type sqlite -o $OUT/stepwall.sqlite $OUT/stepwall.nsys-rep > $OUT/export.log 2>&1 \
  && echo "SQLITE OK" || echo "SQLITE_FAILED (rescue later; rep still queryable)"
head -25 $OUT/kern_sum.txt 2>/dev/null

echo "=== restart service $(date) ==="
# unit ordering now guarantees the daemon gate (Requires+After + readiness
# ExecStartPost) — a bare `start` is safe since 2026-10-14
systemctl start sm70-serve
OK=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "HEALTH_OK after ~$((i*10))s"
    OK=1
    break
  fi
  sleep 10
done
[ $OK -eq 1 ] || echo "FATAL: health not OK after 1200s"
echo "=== window end $(date) ==="
exit 0
