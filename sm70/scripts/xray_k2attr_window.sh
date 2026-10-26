#!/bin/bash
# Consolidated GPU window (audit 2026-10-15):
#  P1: K2 v2 G2-FAIL attribution — per-kernel profiler breakdown (quiet GPU, no service)
#  P2: decode X-ray — nsys --cuda-graph-trace=node on serve (PRODUCTION ENV
#      REPLICATED: AR hier+file+wire0 / QSA FMA+PLE / MoE wide-256) + ruler 65K
#  P3: stats + sqlite + family shares (B1: decode AR share now / B2: MoE share now)
#  P4: restart + health + battery + daemon liveness
#
# Discipline #11928: NO blind sleeps — every wait is a bounded condition poll;
# every phase has precondition + postcondition gates; any unexpected state
# ABORTS (kill strays -> verify GPUs drained -> restore service -> health -> exit 3).
set -uo pipefail
OUT=/data/sm70_results/round5/xray_k2attr
mkdir -p $OUT
cd $OUT

log() { echo "[$(date +%H:%M:%S)] $*"; }
ABORTED=0
restore() {
  # idempotent: kill window strays, verify GPUs drained, service up + healthy
  ABORTED=1
  local floor="${FLOOR:-999999}"
  pkill -9 -f "bench_gdn_k2v2_attr.py" 2>/dev/null || true
  # escaped daemonized serve: vLLM's APIServer setsid's itself (PPID 1,
  # own session) and survives group kills — the run6 stray held :8001 and
  # answered the health check, producing a FALSE GREEN while the unit was
  # failing. Only touch when the unit is not active: during a window,
  # production was stopped in phase0a, so any serve alive here belongs to
  # this window. The daemon runs "preload", not "serve" — pattern-safe.
  if ! systemctl is-active -q sm70-serve; then
    for p in $(pgrep -f "vllm.entrypoints.cli.main serve" || true); do
      kill -9 "$p" 2>/dev/null || true
    done
    # worker/engine strays hold GPU memory and GPU-drain stalls the boot
    # (07:23/07:49 evidence: engine core init failed while old workers
    # drained). The daemon (preload) has no Worker_TP/EngineCore children.
    for p in $(pgrep -f "Worker_TP|VLLM::EngineCore" || true); do
      kill -9 "$p" 2>/dev/null || true
    done
    # nsys agents daemonize too and busy-loop at ~77% CPU (run6/run7
    # strays 2766264/2774168) — and hold CUPTI state
    for p in $(pgrep -f "nsys --start-agent" || true); do
      kill -9 "$p" 2>/dev/null || true
    done
    sleep 3
  fi
  if [ -n "${NSYS_PGID:-}" ] && kill -0 -- -"$NSYS_PGID" 2>/dev/null; then
    kill -INT -- -"$NSYS_PGID" 2>/dev/null; sleep 5
    kill -9 -- -"$NSYS_PGID" 2>/dev/null
  fi
  local i mem
  for i in $(seq 1 24); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$mem" -le "$floor" ] && break
    sleep 2
  done
  if ! systemctl is-active -q sm70-serve; then
    systemctl start sm70-serve
  fi
  for i in $(seq 1 120); do
    # unit-active half of the health gate: a stray APIServer can answer the
    # health probe while the unit is failing (run6 false-green evidence)
    curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && \
      systemctl is-active -q sm70-serve && { log "service healthy"; return 0; }
    sleep 5
  done
  log "RESTORE-FAIL: service not healthy after 10 min"
  return 1
}
abort() { log "ABORT: $*"; restore; exit 3; }
trap restore EXIT

# ---------------------------------------------------------------- baseline
DAEMON_PIDS=$(pgrep -f "vllm.entrypoints.cli.main preload" || true)
[ -n "$DAEMON_PIDS" ] || { log "FATAL: no daemon"; exit 1; }
log "daemon baseline pids: $DAEMON_PIDS"
FLOOR=$(( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) + 500 ))
log "GPU mem floor ${FLOOR}MiB (pre-stop baseline +500)"

# ---------------------------------------------------------------- phase 0
systemctl stop sm70-serve
for i in $(seq 1 60); do
  systemctl is-active -q sm70-serve || break
  sleep 2
done
systemctl is-active -q sm70-serve && abort "service still active 120s after stop"
log "phase0a: service stopped"
for i in $(seq 1 40); do
  age=$(python3 - <<'PY'
import re, subprocess
from datetime import datetime
try:
    out = subprocess.run(
        ["journalctl", "-u", "sm70-serve", "-n", "1", "--output=short-iso"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    ts = out.splitlines()[-1].split(" ", 1)[0]
    ts = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", ts)
    now = datetime.now().astimezone()
    ts_dt = datetime.fromisoformat(ts)
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=now.tzinfo)
    print(int((now - ts_dt).total_seconds()))
except Exception:
    print(-1)
PY
  )
  [ "${age:--1}" -ge 60 ] && break
  sleep 3
done
[ "${age:--1}" -ge 60 ] || abort "journal never reached 60s silence in 120s"
log "phase0b: journal silent ${age}s"
for i in $(seq 1 24); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$mem" -le "$FLOOR" ] && break
  sleep 2
done
[ "$mem" -le "$FLOOR" ] || abort "GPUs not drained to ${FLOOR} (still ${mem})"
log "phase0c: GPUs drained (max ${mem}MiB)"

# ---------------------------------------------------------------- phase 1
log "phase1: K2 v2 attribution bench (timeout 600s)"
timeout 600 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/bench_gdn_k2v2_attr.py > k2v2_attr.log 2>&1
K2RC=$?
grep -q "TOTAL" k2v2_attr.log || abort "k2v2 attr: no TOTAL line (rc=$K2RC)"
[ "$K2RC" -eq 0 ] || abort "k2v2 attr rc=$K2RC"
tail -20 k2v2_attr.log
log "phase1 done: K2RC=0"

# ---------------------------------------------------------------- phase 2
log "phase2: nsys boot $(date '+%F %T') (--cuda-graph-trace=node, prod env replicated)"
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
# prod drop-ins (must match /etc/systemd/system/sm70-serve.service.d/*.conf)
export SM70_QSA_FMA=1
export VLLM_PLE_QUANT_DIR=/data/models/Qwen3.8-Flash-Next-NVFP4/ple-sidecar/ples_nvfp4
export VLLM_SM70_MOE_QPN2_WIDE=auto
export VLLM_SM70_AR_HIER=1
export VLLM_SM70_AR_HIER_FILE=/tmp/ar_hier_mode
export VLLM_SM70_AR_WIRE=0
PGID_FILE=$OUT/xray.pgid
rm -f "$PGID_FILE"
# stale artifacts from earlier runs poison this phase (run3 evidence:
# 0-byte rep left by a SIGKILL'd nsys, stale sqlite made `nsys export`
# fail with "File already exists")
rm -f "$OUT/xray.nsys-rep" "$OUT/xray.sqlite" "$OUT/kern_sum.txt" "$OUT/export.log"
# nsys temp sessions land in /tmp/nvidia (root FS!): run6's session left
# ~66G and filled the 198G root disk, which killed every later serve boot
# (engine workers can't write) and spun the journal. All sessions are dead
# by phase0 (no nsys processes) — sweep stale quadd sessions.
rm -rf /tmp/nvidia/nsight_systems/quadd_session_* 2>/dev/null || true
setsid bash -c "echo \$\$ > $PGID_FILE; exec nsys profile --cuda-graph-trace=node --wait=primary -o $OUT/xray --force-overwrite=true /root/run/run_sm70_serve_ipc.sh" \
  > $OUT/boot.log 2>&1 &
for i in $(seq 1 20); do [ -s "$PGID_FILE" ] && break; sleep 0.5; done
NSYS_PGID=$(cat "$PGID_FILE" 2>/dev/null)
[ -n "$NSYS_PGID" ] || abort "nsys tree not tracked (no pgid)"
log "nsys session pgid=$NSYS_PGID"
READY=0
for i in $(seq 1 150); do
  curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && { READY=1; log "READY after ~$((i*10))s"; break; }
  sleep 10
done
[ "$READY" -eq 1 ] || abort "nsys boot not ready in 1500s"
cd /data/src/vllm-v2
timeout 900 /data/vllm-dev/bin/python sm70_tests/baseline.py --lens 65536 > $OUT/ruler.log 2>&1
RULER_RC=$?
[ "$RULER_RC" -eq 0 ] || abort "ruler rc=$RULER_RC"
grep -E "timed|warm" $OUT/ruler.log
log "phase2 done: ruler OK"

# ---------------------------------------------------------------- phase 3
log "phase3: nsys stop (targets still alive) + stats"
# `nsys stop` = tell the RUNNING session to end capture and write the rep.
# It MUST run while all targets are still alive: the CUPTI final flush
# needs live subscribers (run8 evidence: SIGKILL'ing the workers first
# left the nsys CLI+agent deadlocked on futex, waiting for a flush that
# never comes). The stop client blocks until the report is written —
# 600s covers a ~3min capture (run3's 5.78M-kernel rep took minutes).
timeout 600 nsys stop > "$OUT/nsys_stop.log" 2>&1 \
  || log "WARN nsys stop client rc=$? (see nsys_stop.log)"
# wait for nsys to exit (it does so after writing the rep)
for i in $(seq 1 60); do
  kill -0 "$NSYS_PGID" 2>/dev/null || break
  sleep 5
done
if kill -0 "$NSYS_PGID" 2>/dev/null; then
  kill -9 "$NSYS_PGID" 2>/dev/null
  log "WARN nsys alive 300s after stop request; killed"
fi
[ -s "$OUT/xray.nsys-rep" ] || {
  log "nsys_stop.log tail: $(tail -3 "$OUT/nsys_stop.log" 2>/dev/null | tr '\n' ' ')"
  abort "xray.nsys-rep missing/empty"
}
log "rep written: $(stat -c %s "$OUT/xray.nsys-rep") bytes"
# capture is on disk — NOW tear down the serve tree: TERM, then SIGKILL
# stragglers (vLLM graceful shutdown can hang on leaked shared_memory)
for p in $(pgrep -g "$NSYS_PGID" 2>/dev/null | grep -v "^${NSYS_PGID}$"); do
  kill -TERM "$p" 2>/dev/null || true
done
for i in $(seq 1 24); do
  pgrep -f "Worker_TP|VLLM::EngineCore|run_sm70_serve_ipc" >/dev/null || break
  sleep 5
done
for p in $(pgrep -g "$NSYS_PGID" 2>/dev/null | grep -v "^${NSYS_PGID}$"); do
  kill -9 "$p" 2>/dev/null || true
done
# escaped processes: vLLM's APIServer self-daemonizes (PPID 1) and the
# workers are spawned with renamed cmdlines ("VLLM::Worker_TP0" — the
# "cli.main serve" pattern alone misses them; run6 APIServer / run8
# workers evidence). Production was stopped in phase0a, so anything
# matching now is from this window. The daemon runs "preload".
for p in $(pgrep -f "vllm.entrypoints.cli.main serve|Worker_TP|VLLM::EngineCore" || true); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 5
SRV_LEFT=$(pgrep -f "vllm.entrypoints.cli.main serve|Worker_TP|VLLM::EngineCore" || true)
[ -z "$SRV_LEFT" ] || abort "stray serve pids: $SRV_LEFT"
DAEMON_DEAD=""
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || DAEMON_DEAD="$DAEMON_DEAD $p"
done
[ -z "$DAEMON_DEAD" ] || abort "PROTOCOL VIOLATION: daemon pids dead:$DAEMON_DEAD"
[ -s "$OUT/xray.nsys-rep" ] || abort "xray.nsys-rep missing/empty"
timeout 300 nsys stats --report cuda_gpu_kern_sum $OUT/xray.nsys-rep > $OUT/kern_sum.txt 2>&1 \
  || log "WARN kern_sum failed (rep may be partial)"
# nsys stats already auto-converts the rep to the companion xray.sqlite
# (06:47 evidence: stats wrote 626MB sqlite, export then failed on
# "File already exists"); --force-overwrite makes the export idempotent.
timeout 600 nsys export --type sqlite --force-overwrite true -o $OUT/xray.sqlite $OUT/xray.nsys-rep > $OUT/export.log 2>&1 \
  || abort "sqlite export failed"
timeout 600 /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/family_shares.py --out $OUT > $OUT/family_run.log 2>&1 \
  || abort "family_shares failed (see family_run.log)"
head -35 $OUT/family_shares.md
log "phase3 done: stats + family shares"

# ---------------------------------------------------------------- phase 4
systemctl is-active -q sm70-serve || systemctl start sm70-serve
for i in $(seq 1 120); do
  # unit-active half: a stray APIServer answering the probe while the unit
  # is failing would false-pass (run6 evidence)
  curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && \
    systemctl is-active -q sm70-serve && break
  sleep 5
done
curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && \
  systemctl is-active -q sm70-serve || abort "service not healthy after restart"
sleep 10  # settle before battery (bounded)
export $(grep -oE 'SM70_API_KEY=[0-9a-f]{32}' /data/src/sm70_env.sh)
timeout 1800 /data/vllm-dev/bin/python \
  /data/src/vllm-v2/sm70_tests/check_online_content.py > battery.log 2>&1
BATTERY_RC=$?
tail -2 battery.log
[ "$BATTERY_RC" -eq 0 ] || abort "post-window battery failed (rc=$BATTERY_RC)"
for p in $DAEMON_PIDS; do
  kill -0 "$p" 2>/dev/null || { log "FATAL: daemon pid $p dead"; exit 4; }
done
log "WINDOW OK: K2RC=0 RULER_RC=0 BATTERY_RC=0 daemon untouched"
exit 0
