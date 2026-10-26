#!/bin/bash
# run_checked.sh — gate that refuses to run a script containing the known
# operational footguns of this box.
#
# Rationale: writing the rules into a 15k-line markdown file does not stop me
# from repeating them, because nothing fires at the moment of action. This is
# the gate: it inspects the script, refuses on a hard violation, and only then
# executes.
#
#   /root/run/run_checked.sh <script> [args...]
#
# Exit 2 = refused (nothing executed). Exit code otherwise = the script's.
set -uo pipefail
PREFLIGHT_ONLY=0
if [ "${1:-}" = "--preflight-only" ]; then
  PREFLIGHT_ONLY=1
  shift
fi
SCRIPT="${1:?usage: run_checked.sh [--preflight-only] <script> [args...]}"
shift || true

fail=0
note () { printf '  [%s] %s\n' "$1" "$2"; }

echo "=== preflight: $SCRIPT ==="

# 1) pgrep self-match: a pattern that also appears literally in the script can
#    match the shell running it (it happened twice: killed my own job).
while IFS= read -r pat; do
  [ -z "$pat" ] && continue
  case "$pat" in
    *"["*"]"*) continue ;;   # bracket trick already used
  esac
  if [ "$(grep -c -- "$pat" "$SCRIPT")" -gt 1 ]; then
    note FAIL "pgrep pattern '$pat' also appears elsewhere in the script -> self-match risk (use 'x_[y]' form)"
    fail=1
  fi
done < <(grep -oE 'pgrep -[a-zA-Z]*f[[:space:]]+"[^"]+"' "$SCRIPT" | sed -E 's/.*"([^"]+)".*/\1/')

# 2) systemctl restart of the serve: returns while the old control group dies.
if grep -qE 'systemctl[[:space:]]+restart[[:space:]]+sm70-serve' "$SCRIPT"; then
  note FAIL "systemctl restart sm70-serve -> use /root/run/serve_restart_safe.sh"
  fail=1
fi

# 3) readiness by a sliding journal window (matched an old READY line).
if grep -qE 'journalctl.*--since[[:space:]]+"[0-9]+ min ago"' "$SCRIPT" \
   && grep -qE 'Weight cache daemon READY' "$SCRIPT"; then
  note FAIL "daemon readiness with a relative --since window -> anchor on the start timestamp"
  fail=1
fi

# 4) backups into /tmp (tmpfs).
if grep -qE 'cp[[:space:]].*[[:space:]]/tmp/' "$SCRIPT"; then
  note FAIL "backup copied to /tmp (tmpfs) -> use /data/sm70_results/artifacts"
  fail=1
fi

# 5) daemon restart when the change is serve-only: allowed but must be explicit.
if grep -qE 'serve_cycle_clean\.sh' "$SCRIPT" \
   && ! grep -qiE 'daemon (change|required|needed)|IPC|preload' "$SCRIPT"; then
  note NOTE "full daemon cycle with no stated daemon-side reason (serve-only changes need serve_restart_safe.sh)"
fi

if [ "$fail" -ne 0 ]; then
  echo "=== REFUSED: fix the items above ==="
  exit 2
fi
if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
  echo "=== preflight OK (dry run, nothing executed) ==="
  exit 0
fi
echo "=== preflight OK, running ==="
exec "$SCRIPT" "$@"
