#!/usr/bin/env python
"""Memory/VRAM breakdown for the SM70 stack (read-only).

Answers "who holds what, and which copy is redundant", which is the first step of
the memory item (todo 277): the daemon peaked at 116-125 GB against 123 GB of RAM
(11.6 GB of swap), the engine sits at ~26 GB, /dev/shm carries 27 GB, and the PLE
table alone is ~26 GB of payload -- so the question is how many copies exist.

Usage: mem_report.py [--json] [tag]
Writes /data/sm70_results/mem/<tag>.txt.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import time

UNITS = ("sm70-serve", "sm70-daemon")


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def unit_pids(unit: str) -> list[int]:
    out = sh(["systemctl", "show", unit, "-p", "MainPID", "--value"]).strip()
    pids = [int(out)] if out.isdigit() and int(out) else []
    # children (workers) live in the same cgroup
    cg = sh(["systemctl", "show", unit, "-p", "ControlGroup", "--value"]).strip()
    if cg:
        for path in pathlib.Path("/sys/fs/cgroup" + cg).rglob("cgroup.procs"):
            for line in path.read_text().splitlines():
                if line.strip().isdigit():
                    pids.append(int(line))
    return sorted(set(pids))


def rollup(pid: int) -> dict[str, int]:
    try:
        text = pathlib.Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return {}
    vals: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(\w+(?:_\w+)?):\s+(\d+) kB", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) * 1024
    return vals


def _classify(pathname: str) -> str:
    """Bucket one smaps mapping by what it is."""
    if not pathname:
        return "anon"
    if pathname.startswith("/dev/shm/"):
        return "shm"
    if pathname.startswith(("/dev/nvidia", "/proc/driver/nvidia")):
        return "cuda_host"
    if pathname.startswith(("[", "/memfd", "/dev/zero")):
        return "anon" if pathname in ("[heap]",) or pathname.startswith("[anon") else "special"
    if pathname.startswith(("/usr/", "/data/vllm-dev", "/data/venv-sync", "/data/src")):
        return "file_lib"
    return "file_other"


def mappings_report(unit: str, top: int = 12) -> list[str]:
    """Aggregate one unit's smaps by mapping class (Pss) and list the top ones.

    Answers "what is the daemon's 24.8 GB Pss?" -- which the per-process rollup
    cannot: anonymous weight copies, /dev/shm PLE shards and CUDA host
    allocations all look the same there.
    """
    out: list[str] = []
    agg: dict[str, dict[str, int]] = {}
    tops: list[tuple[int, str, str]] = []
    for pid in unit_pids(unit):
        try:
            lines = pathlib.Path(f"/proc/{pid}/smaps").read_text().splitlines()
        except OSError:
            continue
        cur: str | None = None
        for line in lines:
            if "-" in line and line.split()[0].count("-") == 1 and " " in line and not line.startswith(("Size", "Rss", "Pss", "Vm")):
                parts = line.split()
                if len(parts) >= 5:
                    cur = parts[5] if len(parts) > 5 else ""
                continue
            if line.startswith("Pss:"):
                kb = int(line.split()[1])
                cls = _classify(cur or "")
                slot = agg.setdefault(cls, {"pss": 0, "n": 0})
                slot["pss"] += kb * 1024
                slot["n"] += 1
                if kb * 1024 > (32 << 20):
                    tops.append((kb * 1024, cls, (cur or "<anon>")[:60]))
    if not agg:
        return out
    total = sum(v["pss"] for v in agg.values())
    out.append(f"\n[{unit}] mappings (Pss, {total/2**30:.1f} GB total)")
    for cls, v in sorted(agg.items(), key=lambda kv: -kv[1]["pss"]):
        out.append(f"  {cls:11s} {v['pss']/2**30:7.2f} GB  ({v['n']} mappings)")
    out.append("  -- top mappings > 32 MB")
    for sz, cls, name in sorted(tops, reverse=True)[:top]:
        out.append(f"    {sz/2**30:6.2f} GB  {cls:10s} {name}")
    return out


def main() -> int:
    tag = next((a for a in sys.argv[1:] if not a.startswith("-")), time.strftime("%m%d-%H%M"))
    out = pathlib.Path("/data/sm70_results/mem")
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say(f"== SM70 memory report {tag} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ==")

    say("\n[host]")
    free = sh(["free", "-m"]).splitlines()
    for line in free[:3]:
        say("  " + "  ".join(line.split()[:5]))

    for unit in UNITS:
        cur = sh(["systemctl", "show", unit, "-p", "MemoryCurrent", "--value"]).strip()
        peak = sh(["systemctl", "show", unit, "-p", "MemoryPeak", "--value"]).strip()
        active = sh(["systemctl", "is-active", unit]).strip()

        def gb(v: str) -> str:
            return f"{int(v)/2**30:.1f} GB" if v.isdigit() else v

        say(f"\n[{unit}] state={active} current={gb(cur)} peak={gb(peak)}")
        agg = {"Pss": 0, "Private_Clean": 0, "Private_Dirty": 0,
               "Shared_Clean": 0, "Shared_Dirty": 0, "Swap": 0}
        rows = []
        for pid in unit_pids(unit):
            r = rollup(pid)
            if not r:
                continue
            rows.append((pid, r))
            for k in agg:
                agg[k] += r.get(k, 0)
        rows.sort(key=lambda t: -t[1].get("Pss", 0))
        say("  pid      Pss   priv_dirty  shared   swap   cmd")
        for pid, r in rows[:10]:
            cmd = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")[:60]
            say(f"  {pid:<8} {r.get('Pss',0)/2**30:5.1f}G "
                f"{r.get('Private_Dirty',0)/2**30:8.1f}G "
                f"{(r.get('Shared_Clean',0)+r.get('Shared_Dirty',0))/2**30:6.1f}G "
                f"{r.get('Swap',0)/2**30:5.1f}G  {cmd}")
        say("  ---- totals: " + "  ".join(f"{k}={v/2**30:.1f}G" for k, v in agg.items()))

    say("\n[/dev/shm]")
    total = 0
    files = []
    for p in pathlib.Path("/dev/shm").iterdir():
        try:
            sz = p.stat().st_size if p.is_file() else 0
        except OSError:
            continue
        total += sz
        if sz:
            files.append((sz, p.name))
    for sz, name in sorted(files, reverse=True)[:12]:
        say(f"  {sz/2**30:7.2f} GB  {name}")
    say(f"  ---- {len(files)} files, {total/2**30:.1f} GB")

    say("\n[GPU]")
    for row in sh(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                   "--format=csv,noheader"]).splitlines():
        say("  " + row)

    if "--mappings" in sys.argv:
        for unit in UNITS:
            for line in mappings_report(unit):
                say(line)

    say("\n[PLE payload for reference] 320,001,600 rows x dim160: codes 80 B/row "
        "+ per-16 scales 10 B/row ~= 26 GB per full copy")

    (out / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwritten to {out / (tag + '.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
