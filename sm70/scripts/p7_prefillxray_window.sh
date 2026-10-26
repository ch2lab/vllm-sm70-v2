#!/bin/bash
# P7 prefill X-ray window: nsys on serve with FULL production env,
# workload = 3 fresh 65K prefills (max_tokens=1, APC-immune salts).
# Profile is >99.8% prefill kernel time -> kernel family table = new shares
# after the MoE wide-256 change.
#
# Window env MUST mirror production drop-ins (lesson #11886):
#   SM70_QSA_FMA=1, VLLM_PLE_QUANT_DIR, VLLM_SM70_MOE_QPN2_WIDE=auto.
# Safe-fallback + daemon read-only protocol = stepwall_window.sh (P5).
OUT=/data/sm70_results/round5/p7_prefillxray
mkdir -p $OUT
cd $OUT

# --- long-lived component bookkeeping (read-only baseline) -----------------
DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
echo "daemon pids (read-only baseline): ${DAEMON_PIDS:-NONE}"
[ -n "$DAEMON_PIDS" ] || { echo "FATAL: no daemon running before window — refuse to start"; exit 1; }

cleanup() {
  if ! systemctl is-active -q sm70-serve; then
    echo "cleanup: service not active -> restarting"
    systemctl start sm70-serve
  fi
}
trap cleanup EXIT

echo "=== window start $(date) ==="
systemctl stop sm70-serve
sleep 5

export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
export SM70_QSA_FMA=1
export VLLM_PLE_QUANT_DIR=/data/models/Qwen3.8-Flash-Next-NVFP4/ple-sidecar/ples_nvfp4
export VLLM_SM70_MOE_QPN2_WIDE=auto   # production drop-in mirror

echo "=== nsys boot $(date) ==="
PGID_FILE=$OUT/nsys.pgid
rm -f "$PGID_FILE"
setsid bash -c "echo \$\$ > $PGID_FILE; exec nsys profile -o $OUT/prefillxray --force-overwrite=true /root/run/run_sm70_serve_ipc.sh" \
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

echo "=== prefill workload (3x 65K, max_tokens=1) $(date) ==="
cd /data/src/vllm-v2
timeout 900 /data/vllm-dev/bin/python sm70_tests/p7_prefill_only.py 3 > $OUT/prefill_workload.log 2>&1
WRC=$?
echo "WORKLOAD_RC=$WRC"
cat $OUT/prefill_workload.log

echo "=== nsys stop $(date) ==="
kill -INT -- -"$NSYS_PID" 2>/dev/null
for i in $(seq 1 15); do
  kill -0 -- -"$NSYS_PID" 2>/dev/null || break
  sleep 10
done
kill -9 -- -"$NSYS_PID" 2>/dev/null   # rest of the window tree (serve + workers)
sleep 5
SRV_LEFT=$(pgrep -f "vllm.entrypoints.cli.main serve" || true)
[ -n "$SRV_LEFT" ] && echo "FATAL: stray serve pids: $SRV_LEFT" || echo "no stray serve processes"
DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -n "$DAEMON_DEAD" ] && echo "FATAL: PROTOCOL VIOLATION — daemon pids dead:$DAEMON_DEAD" || echo "daemon liveness OK (untouched)"
echo "nsys exited"

ls -la $OUT/prefillxray.nsys-rep 2>/dev/null
timeout 300 nsys stats --report cuda_gpu_kern_sum $OUT/prefillxray.nsys-rep > $OUT/kern_sum.txt 2>/dev/null \
  && echo "STATS OK" || echo "STATS_FAILED (rescue later)"
timeout 600 nsys export --type sqlite -o $OUT/prefillxray.sqlite $OUT/prefillxray.nsys-rep > $OUT/export.log 2>&1 \
  && echo "SQLITE OK" || echo "SQLITE_FAILED (rescue later)"
head -30 $OUT/kern_sum.txt 2>/dev/null

echo "=== restart service $(date) ==="
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
