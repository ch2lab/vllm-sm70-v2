#!/usr/bin/env python3
"""Decode X-ray family shares (audit items B1/B2: NCCL-AR now? MoE now?).

Inputs (all in --out dir):
  xray.sqlite  — nsys export (--cuda-graph-trace=node: graph kernels appear
                 as individual CUPTI records)
  window.log   — '=== nsys boot <date %F %T> ===' line = trace origin
  /data/sm70_results/baseline_window/baseline_<YYYY-MM-DD>.jsonl — ruler rows
                 (warm/timed1/timed2 @65536 with perf.prefill_ms), filtered to
                 rows with ts >= window start

Phase segmentation = P5 phase_split convention (dispatch ts + prefill_ms
calibrated against trace origin). Outputs markdown:
  * per-phase (PREFILL_*/DECODE_*) family shares + kernel-time totals
  * raw top-30 kernels per DECODE phase (to VERIFY the family regex table
    against real names — evidence before classification)
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict

FAMILIES = [
    ("AR", re.compile(r"nccl|sm70_hier_ar|allreduce|reduce_scatter", re.I)),
    ("MOE", re.compile(r"nvfp4_qpn2|wmma|_moe|gated_fused", re.I)),
    ("GDN", re.compile(r"gdn|chunk_gated_delta|mamba", re.I)),
    ("QSA", re.compile(r"qsa|prefill_fma", re.I)),
    ("DENSE", re.compile(r"gemm|gemv|sgemm|hgemm|cublas", re.I)),
    ("ATT", re.compile(r"flash|fmha|attn", re.I)),
]


def family_of(name: str) -> str:
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return "OTHER"


def parse_local(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    # --- trace origin from the newest window_run*.log (nsys boot line) ---
    import glob as _glob
    import os as _os
    wlogs = sorted(_glob.glob(f"{args.out}/window_run*.log"),
                   key=lambda p: _os.path.getmtime(p)) or [f"{args.out}/window.log"]
    origin = None
    for wlg in wlogs:
        for line in open(wlg):
            m = re.search(r"nsys boot (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                origin = parse_local(m.group(1))
                break
        if origin:
            break
    assert origin, "nsys boot line not found in window_run*.log"

    # --- ruler rows for THIS window ---
    day = time.strftime("%Y-%m-%d")
    jsonl = args.jsonl or f"/data/sm70_results/baseline_window/baseline_{day}.jsonl"
    start_s = origin - 5.0
    runs = []
    for line in open(jsonl):
        r = json.loads(line)
        if r.get("length") == 65536 and parse_local(r["ts"]) >= start_s:
            runs.append({"tag": r["run"], "ts": parse_local(r["ts"]),
                         "pf_ms": (r.get("perf") or {}).get("prefill_ms") or 0.0,
                         # jsonl `ts` = request COMPLETION (after decode);
                         # dec_ms null for timed runs (client does not wait)
                         "dec_ms": (r.get("perf") or {}).get("decode_ms") or 0.0})
    order = {"warm": 0, "timed1": 1, "timed2": 2}
    runs = [r for r in runs if r["tag"] in order]
    runs.sort(key=lambda r: order[r["tag"]])
    assert len(runs) == 3, f"expected 3 ruler rows, got {len(runs)}"

    db = sqlite3.connect(f"{args.out}/xray.sqlite")
    kmax = db.execute(
        "SELECT MAX(end), MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    rows = db.execute("""SELECT k.start, k.end, s.value
                         FROM CUPTI_ACTIVITY_KIND_KERNEL k
                         LEFT JOIN StringIds s ON k.demangledName = s.id
                         """).fetchall()

    # --- anchor the CUPTI timeline to wall clock ---
    # CUPTI timestamps are relative to the profiling session, NOT to the
    # "nsys boot" wall time (evidence run5: ~3s drift, and a stale sqlite
    # even carried the WRONG session). Anchor = first dense 16s CUPTI
    # burst, which is the warm PREFILL (ruler jsonl gives its wall start).
    session_start = db.execute(
        "SELECT localTime FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    kmin, kmaxv = kmax[1], kmax[0]
    nbin = int((kmaxv - kmin) // 5e8) + 2
    dens = [0] * nbin
    for s, _e, _n in rows:
        dens[int((s - kmin) // 5e8)] += 1
    warm = next(r for r in runs if r["tag"] == "warm")
    warm_pf0 = warm["ts"] - warm["dec_ms"] / 1000.0 - warm["pf_ms"] / 1000.0
    # Expected CUPTI position of warm prefill from the session-start wall
    # time. Dense 16s windows elsewhere (boot-time CUDA-graph-capture
    # bursts are just as dense) must NOT win the anchor -> pick the dense
    # window NEAREST the expected position.
    ss_epoch = time.mktime(time.strptime(session_start.replace("T", " "),
                                         "%Y-%m-%d %H:%M:%S"))
    exp_warm_cu = warm_pf0 - ss_epoch
    # Jump-based anchor: first 0.5s bin whose kernel count jumps to
    # prefill density (>2500; boot/graph-capture bursts are outside the
    # +/-15s expectation window, the sparse pre-prefill ramp is <1200).
    lo = max(0, int(((exp_warm_cu - 15.0) * 1e9 - kmin) // 5e8))
    hi = min(nbin, int(((exp_warm_cu + 15.0) * 1e9 - kmin) // 5e8) + 1)
    anchor_cu = None
    for i in range(lo, hi):
        if dens[i] > 2500:
            anchor_cu = (kmin + i * 5e8) / 1e9
            break
    assert anchor_cu is not None, "no prefill density jump near expected position"
    SHIFT = warm_pf0 - anchor_cu  # kernel_wall_epoch = cu_rel + SHIFT
    print(f"kernels: {len(rows)}; session={session_start} "
          f"span {kmin/1e9 + SHIFT - origin:.1f}s .. "
          f"{kmaxv/1e9 + SHIFT - origin:.1f}s (window-rel)")

    trace_end = kmaxv / 1e9 + SHIFT  # wall epoch
    windows = []
    for i, r in enumerate(runs):
        # jsonl ts = completion; ttft = ts - decode; prefill = [ttft-pf, ttft]
        ttft = r["ts"] - r["dec_ms"] / 1000.0
        pf0 = ttft - r["pf_ms"] / 1000.0
        windows.append((f"PREFILL_{r['tag']}", pf0 - 0.5, ttft + 0.5))
        if i + 1 < len(runs):
            nxt = (runs[i + 1]["ts"] - runs[i + 1]["dec_ms"] / 1000.0
                   - runs[i + 1]["pf_ms"] / 1000.0) - 0.5
        else:
            nxt = None
        # DECODE_{tag} = from this ttft until the next request's prefill
        # starts (captures this request's decode + any idle)
        windows.append((f"DECODE_{r['tag']}", ttft + 0.5,
                        nxt if nxt is not None else trace_end))

    def win_of(s: float):
        for name, a, b in windows:
            if a <= s < b:
                return name
        return None

    tot = defaultdict(float)
    fam_tot = defaultdict(lambda: defaultdict(float))
    raw = defaultdict(lambda: defaultdict(float))
    for start, end, name in rows:
        s, e = start / 1e9 + SHIFT, end / 1e9 + SHIFT
        w = win_of(s)
        if w is None:
            continue
        dur = e - s
        tot[w] += dur
        fam_tot[w][family_of(name or "?")] += dur
        if w.startswith("DECODE"):
            raw[w][name or "?"] += dur

    md = [f"# decode X-ray family shares — {time.strftime('%F %T')}",
          "trace: nsys --cuda-graph-trace=node on serve (prod env replicated: "
          "AR hier+wire0 / QSA FMA / MoE wide-256) | ruler 65536 warm+timed1+timed2", ""]
    for w in [n for n, _, _ in windows]:
        t = tot[w]
        md.append(f"## {w} — kernel time {t*1000:.1f}ms")
        md.append("| family | ms | % |")
        md.append("|--------|----|---|")
        for fam, ms in sorted(fam_tot[w].items(), key=lambda kv: -kv[1]):
            md.append(f"| {fam} | {ms*1000:.1f} | {100*ms/t:.1f}% |")
        md.append("")
    for w in [n for n, _, _ in windows if n.startswith("DECODE")]:
        t = tot[w]
        md.append(f"## raw top-30 {w}")
        md.append("| kernel | ms | % |")
        md.append("|--------|----|---|")
        for name, ms in sorted(raw[w].items(), key=lambda kv: -kv[1])[:30]:
            md.append(f"| `{name[:70]}` | {ms*1000:.1f} | {100*ms/t:.1f}% |")
        md.append("")
    out = "\n".join(md)
    print(out)
    with open(f"{args.out}/family_shares.md", "w") as fh:
        fh.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
