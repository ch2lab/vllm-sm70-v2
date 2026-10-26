#!/bin/bash
# fp4local_switch.sh on|off|status
#
# sm70/08 §5.3 data-plane valve: point the preload daemon and the serve at the
# FP4-Local-NVFP4 checkpoint and drop VLLM_PLE_QUANT_DIR (the old sidecar holds
# the OLD table; the new checkpoint carries a pre-quantized NVFP4 PLE in its
# main shard that the passthrough loader reads directly).
#
# Edits ONLY: the launch scripts' model path, the `export VLLM_PLE_QUANT_DIR=`
# lines, and the systemd drop-in's `Environment=VLLM_PLE_QUANT_DIR=` line.
# Every other flag/valve is untouched. Does NOT restart anything: both
# processes read these files only at boot, so nothing changes until T7.
#
# `on` snapshots each file to `<file>.bak_fp4local` (once).
# `off` re-derives what `on` produced from that snapshot and REFUSES unless the
# live file matches it exactly — so it can never clobber an unrelated edit made
# in between, and it restores the deleted env lines byte-exactly.
#
# Testability: FP4LOCAL_RUN_DIR / FP4LOCAL_DROPIN override the paths and
# FP4LOCAL_DRY=1 skips daemon-reload, so a round-trip can be proven on copies.
set -euo pipefail

OLD_MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
NEW_MODEL=/data/models/Qwen3.8-Flash-Next-FP4-Local-NVFP4
RUN_DIR=${FP4LOCAL_RUN_DIR:-/root/run}
DROPIN=${FP4LOCAL_DROPIN:-/etc/systemd/system/sm70-serve.service.d/sm70_fma.conf}
FILES=(
  "$RUN_DIR/run_sm70_preload.sh"
  "$RUN_DIR/run_sm70_serve_ipc.sh"
  "$RUN_DIR/run_sm70_serve.sh"
)

usage() { echo "usage: $0 on|off|status"; exit 2; }

apply_on() {  # $1 = source file, $2 = destination file
  cp "$1" "$2"
  sed -i "s|${OLD_MODEL}|${NEW_MODEL}|g" "$2"
  sed -i '/^export VLLM_PLE_QUANT_DIR=/d' "$2"
  sed -i '/^Environment=VLLM_PLE_QUANT_DIR=/d' "$2"
}

do_on() {
  for f in "${FILES[@]}" "$DROPIN"; do
    [ -f "$f" ] || continue
    [ -f "$f.bak_fp4local" ] || cp "$f" "$f.bak_fp4local"
    apply_on "$f.bak_fp4local" "$f"
    case "$f" in *.sh) bash -n "$f" || { echo "SYNTAX ERROR in $f"; exit 1; } ;; esac
  done
  [ "${FP4LOCAL_DRY:-0}" = "1" ] || systemctl daemon-reload
  echo "ARMED: new checkpoint + no VLLM_PLE_QUANT_DIR. A boot now brings up the NEW model."
}

do_off() {
  armed=0
  for f in "${FILES[@]}" "$DROPIN"; do
    [ -f "$f.bak_fp4local" ] && armed=1
  done
  if [ "$armed" = "0" ]; then
    echo "DISARMED already: no snapshots exist (this valve was never armed)."
    return 0
  fi
  # Verify first, mutate second: every live file must be either already at the
  # snapshot (already disarmed -> no-op) or exactly what `on` produced.
  for f in "${FILES[@]}" "$DROPIN"; do
    [ -f "$f" ] || continue
    if cmp -s "$f.bak_fp4local" "$f"; then
      continue  # already disarmed: idempotent no-op
    fi
    tmp=$(mktemp)
    apply_on "$f.bak_fp4local" "$tmp"
    if ! cmp -s "$tmp" "$f"; then
      rm -f "$tmp"
      echo "REFUSE: $f changed beyond this valve's edits since the snapshot;" >&2
      echo "        reconcile by hand, then re-run." >&2
      exit 1
    fi
    rm -f "$tmp"
  done
  for f in "${FILES[@]}" "$DROPIN"; do
    [ -f "$f" ] || continue
    cp "$f.bak_fp4local" "$f"
    case "$f" in *.sh) bash -n "$f" || { echo "SYNTAX ERROR in $f"; exit 1; } ;; esac
  done
  [ "${FP4LOCAL_DRY:-0}" = "1" ] || systemctl daemon-reload
  echo "DISARMED: old checkpoint + VLLM_PLE_QUANT_DIR restored (byte-exact from snapshot)."
}

do_status() {
  for f in "${FILES[@]}" "$DROPIN"; do
    [ -f "$f" ] || { printf '%-30s (absent)\n' "$(basename "$f")"; continue; }
    new=$(grep -c "$NEW_MODEL" "$f" || true)
    old=$(grep -c "$OLD_MODEL" "$f" || true)
    ple=$(grep -c 'VLLM_PLE_QUANT_DIR=' "$f" || true)
    model=$([ "$new" -gt 0 ] && echo NEW || { [ "$old" -gt 0 ] && echo OLD || echo "-"; })
    printf '%-30s model=%-4s ple_env=%s\n' "$(basename "$f")" "$model" "$ple"
  done
  if [ "${new:-0}" -gt 0 ] 2>/dev/null; then echo "(state: ARMED)"; fi
}

case "${1:-status}" in
  on) do_on ;;
  off) do_off ;;
  status) do_status ;;
  *) usage ;;
esac
