#!/bin/bash
# ram_guard.sh — SOFT host-RAM pre-flight check.
#
# Corrected model (2026-10-18; cgroup memory.stat + /proc/meminfo + vmstat).
# Earlier revisions of this file claimed the box ran at "113/123 GiB committed"
# and refused on swap-in-use.  Both were WRONG:
#   sm70-daemon memory.current 90.8 GiB = anon 6.2 + file 84.1 (weight images
#     mmap'd -> RECLAIMABLE page cache; 27.0 GiB of that file is shmem)
#   sm70-serve  memory.current 22.1 GiB = anon 19.3 + shmem 2.7
#   => non-reclaimable steady state ~61 GiB of 123 GiB (anon ~25 + shmem ~30)
#   => MemAvailable ~61 GiB.  `free`'s used/shared must NOT be added together,
#      and cgroup memory.current must NOT be read as "RAM used".
#   => swap's ~5 GiB is STALE (vmstat si/so = 0), NOT active pressure.
# A serve restart adds ~19 GiB anon, trivially inside ~61 GiB.
# So: this guard is advisory only; the default threshold is low on purpose.
#
# Usage: ram_guard.sh [min_available_gib] [label]
#   exit 0 ok | 3 warn (only when MemAvailable is genuinely low)
MIN="${1:-30}"; LABEL="${2:-step}"
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "${AVAIL:-0}" -lt "$MIN" ]; then
  echo "WARN: ${LABEL}: MemAvailable=${AVAIL}GiB < ${MIN}GiB"
  exit 3
fi
echo "ram_guard OK: ${LABEL}: MemAvailable=${AVAIL}GiB (>= ${MIN})"
exit 0
