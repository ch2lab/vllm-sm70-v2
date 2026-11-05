"""Instruction mix of the non-GEMM attention kernels at BN=8192 / WPR=4.

The power sample says prefill is power-capped (271-273 W of 300 W, pclk
1462-1470 vs 1530 boost) with sm 95.8% but dram only 56%, so the watts come
from the SM array, not HBM. That makes "cut SM-side work" the one lever class
that pays twice - less time AND recovered clock. But it only pays on kernels
that are actually arithmetic-limited; applying it to a bandwidth-bound kernel
buys nothing (proven twice already, laws 6 and the route-tiled arms).

ncu does not support Volta, so this reads the opcode histogram from
cuobjdump -sass instead, and reports each kernel's measured in-service us/call
next to its instruction mix so the two can be cross-read.

Interpretation guide:
  MUFU   = transcendentals (exp/rsqrt) - SFU-pipe pressure
  HMUL2/HADD2/HFMA2 = packed half ops - the e2m1 fold class targets these
  F2F/F16/FMUL/FFMA = scalar float math
  LDG/STG/LDS/STS  = global/shared traffic; if these dominate, it is a
                     bandwidth kernel and instruction cuts are the wrong tool
  PRMT/LOP3/SHF    = the byte-permute dequant path
"""
import re
import subprocess
import sys

SO = sys.argv[1] if len(sys.argv) > 1 else (
    "/root/.cache/torch_extensions/py312_cu128/"
    "sm70_nvfp4_prefill_twopass_v1/sm70_nvfp4_prefill_twopass_v1.so")

# in-service measurement from the BN=8192/WPR=4 arm, for cross-reading
MEASURED = {
    "fused_softmax_reg": (28.13, 20700),   # s, launches -> us/call
    "fused_update": (2.08, 22060),
    "bridge_kv": (1.57, 11800),
}

out = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", SO],
                     capture_output=True, text=True).stdout

# split into per-function blocks
funcs = {}
cur = None
for line in out.splitlines():
    m = re.search(r"Function : (\S+)", line)
    if m:
        cur = m.group(1)
        funcs[cur] = []
        continue
    if cur:
        # A SASS body line looks like:  /*00a0*/   @!P0 LDG.E.64 R4, [R6];
        # The opcode is the first token AFTER any predicate guard.  Stripping a
        # character *set* (the previous lstrip("@!P0123456789 ")) ate the leading
        # P of PRMT into "RMT" and turned predicated instructions into "", so the
        # histogram silently mislabelled the dequant path.
        m = re.match(r"\s+/\*[0-9a-f]{1,}\*/\s+(.+?)\s*[;|]", line)
        if not m:
            continue
        toks = m.group(1).split()
        if toks and toks[0].startswith("@"):     # predicate guard, not an opcode
            toks = toks[1:]
        if not toks:
            continue
        op = toks[0].split(".")[0]
        if re.fullmatch(r"[A-Z][A-Z0-9]*", op):
            funcs[cur].append(op)

CLASS = {
    "MUFU": "SFU(exp/rsqrt)", "HMUL2": "packed-half", "HADD2": "packed-half",
    "HFMA2": "packed-half", "FMUL": "scalar-fp", "FFMA": "scalar-fp",
    "FADD": "scalar-fp", "F2F": "convert", "I2F": "convert", "F2I": "convert",
    "LDG": "global-mem", "STG": "global-mem", "LDS": "shared-mem",
    "STS": "shared-mem", "LD": "mem", "ST": "mem",
    "PRMT": "byte-permute", "LOP3": "logic", "SHF": "shift",
    "IMAD": "int", "IADD3": "int", "MOV": "mov", "SEL": "sel",
    "ISETP": "cmp/br", "FSETP": "cmp/br", "BRA": "cmp/br", "HMMA": "TENSOR-CORE",
}


def demangle(n):
    m = re.search(r"(_Z\w+)", n)
    return (m.group(1) if m else n)[:52]


def short(n):
    if "fused_softmax_reg" in n:
        a = re.findall(r"ILi(\d+)ELi(\d+)", n)
        return f"fused_softmax_reg<{a[0][0]},{a[0][1]}>" if a else "fused_softmax_reg"
    if "fused_update" in n:
        return "fused_update"
    if "bridge_kv" in n:
        return "bridge_kv"
    if "fused_softmax_gen" in n:
        return "fused_softmax_gen"
    return demangle(n)


want = ("fused_softmax_reg", "fused_update", "bridge_kv", "fused_softmax_gen")
seen = {}
for name, ops in funcs.items():
    s = short(name)
    if not s.startswith(want):
        continue
    key = re.sub(r"<.*>", "", s)
    if key in seen:            # keep the largest variant per family
        if len(ops) <= seen[key][1]:
            continue
    seen[key] = (s, len(ops), ops)

print(f"{'kernel':28} {'instr':>6}  {'TOP OPCODES (count, class)'}")
print("-" * 108)
for key, (s, n, ops) in sorted(seen.items(), key=lambda kv: -kv[1][1]):
    from collections import Counter
    c = Counter(ops)
    top = c.most_common(9)
    txt = "  ".join(f"{o}:{k}" for o, k in top)
    print(f"{s:28} {n:6}  {txt}")
    # class rollup
    cls = Counter()
    for o, k in c.items():
        cls[CLASS.get(o, "other")] += k
    tot = sum(cls.values())
    mem = cls["global-mem"] + cls["shared-mem"] + cls["mem"]
    arith = (cls["packed-half"] + cls["scalar-fp"] + cls["int"] + cls["convert"]
             + cls["byte-permute"] + cls["logic"] + cls["shift"])
    meas = MEASURED.get(key)
    tag = ""
    if meas:
        tag = f"   [in-service {meas[0]:.2f} s / {meas[1]} calls = {meas[0]*1e6/meas[1]:.0f} us/call]"
    print(f"{'':28} -> mem {100*mem/tot:4.1f}%   SM-arith {100*arith/tot:4.1f}%"
          f"   SFU {100*cls['SFU(exp/rsqrt)']/tot:4.1f}%   cmp/br {100*cls['cmp/br']/tot:4.1f}%{tag}")
    verdict = ("ARITHMETIC-limited candidate" if arith > 2.0 * mem
               else "bandwidth/mixed - instruction cuts likely the wrong tool")
    print(f"{'':28} => {verdict}")
