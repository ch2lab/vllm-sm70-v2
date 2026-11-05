"""Slice the timestamped power log by the phase boundaries the runner recorded.

Energy is integrated over the ACTUAL sample deltas: `nvidia-smi -lms 250` in
practice emitted one row every ~69 ms here (7620 rows in 526 s), so any fixed
per-sample weight is wrong.  Deriving dt from the timestamps makes it robust to
the driver's real sampling rate.

Answers what time alone cannot: under a 300 W cap that has already pulled pclk
from 1530 to ~1420, energy and speed are coupled, so J/token is reported next
to the wall time.
"""
import csv
import sys
from datetime import datetime, timedelta

out = sys.argv[1] if len(sys.argv) > 1 else "/data/sm70_results/tp1_prof/adopt_bn8192"

rows = []
for r in csv.reader(open(f"{out}/power.csv")):
    if len(r) < 6:
        continue
    try:
        ts = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        pwr = float(r[1].split()[0]); clk = float(r[2].split()[0])
        sm = float(r[3].split()[0]); dram = float(r[4].split()[0])
    except (ValueError, IndexError):
        continue
    rows.append((ts, pwr, clk, sm, dram))
rows.sort(key=lambda x: x[0])
span = (rows[-1][0] - rows[0][0]).total_seconds()
rate = len(rows) / span
print(f"# {len(rows)} samples over {span:.0f} s  ({rate:.1f} Hz actual, "
      f"nominal -lms 250 => {1000/rate:.0f} ms)")


def integ(t0, t1):
    """Trapezoid-free: each sample carries the delta to the next sample."""
    w = [r for r in rows if t0 <= r[0] <= t1]
    if len(w) < 5:
        return None
    e = sum(w[i][1] * ((w[i + 1][0] - w[i][0]).total_seconds()) for i in range(len(w) - 1))
    dur = (w[-1][0] - w[0][0]).total_seconds()
    return dict(n=len(w), dur=dur, E=e, avg=e / dur if dur else 0,
                mx=max(r[1] for r in w), mn=min(r[1] for r in w),
                clk=sorted(r[2] for r in w)[len(w) // 2],
                sm=sum(r[3] for r in w) / len(w),
                dram=sum(r[4] for r in w) / len(w))


def parse(s):
    """'HH:MM:SS[.m(mm)]' -> timedelta from midnight.  The fraction is parsed as
    fractional seconds, not milliseconds: the log carries 3 digits but strftime
    %f carries 6, and int() of either as ms would be wrong by 1000x/100x."""
    h, m, rest = s.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=float(rest))


marks = []
for ln in open(f"{out}/phases.txt"):
    p = ln.split()
    if len(p) < 2:
        continue
    marks.append((parse(p[0]), " ".join(p[1:])))

# pair consecutive start/end (or start/next-start when the label lost "start")
seq = [(m[0], m[1]) for m in marks]          # timedelta-from-midnight, label


def rel(td):
    """timedelta-from-midnight -> absolute sample time, anchored on the log start."""
    return rows[0][0] + td - parse(rows[0][0].strftime("%H:%M:%S.%f"))


# The marks file is chronological and every phase boundary is one line, so a
# phase is simply two consecutive marks; no start/end pairing to get wrong.
windows = [(seq[i][1], rel(seq[i][0]), rel(seq[i + 1][0])) for i in range(len(seq) - 1)]

TOK = {"prefill250k_rep1": 250000, "prefill250k_rep2": 250000, "prefill250k_rep3": 250000,
       "decode65k": 512, "decode10k": 512}
for name, a, b in windows:
    r = integ(a, b)
    if not r:
        continue
    key = name.replace(" start", "")
    jp = f"  {1000*r['E']/TOK[key]:7.2f} J/token" if key in TOK else ""
    print(f"  {name:24} {r['dur']:6.1f} s  pwr avg {r['avg']:6.1f} W "
          f"(min {r['mn']:5.1f} max {r['mx']:5.1f})  pclk {r['clk']:5.0f} MHz  "
          f"sm {r['sm']:5.1f}%  dram {r['dram']:5.1f}%  E {r['E']/1000:6.2f} kJ{jp}")
