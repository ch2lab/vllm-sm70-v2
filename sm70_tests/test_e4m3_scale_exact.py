"""Exhaustive check of the e4m3 group-scale decode used by the nvfp4 bridge.

Why: bridge_kv decodes each 16-dim group's scale with e4m3(), which hand-rolls
the subnormal case (m * 2^-9), the normal case ((1+m/8) * 2^(e-7)) and the 448
sentinel (e==15 && m==7).  Three hand-written branches on a quantisation path
deserve an exhaustive guard rather than a sample.

The technique is borrowed from strata's nvfp4_decode_bench, which opens by
printing "ue4m3 decode: 0 of 256 bytes differ from the ldexpf form" - it refuses
to run at all unless its bit-built scale decode matches the reference over every
one of the 256 byte values.  Adopting the *verification discipline* is lossless;
adopting its W4A8 compute (int8 activations, block_q8_1) would not be, so that
part is deliberately not taken.

Runs on CPU: this checks arithmetic, not kernels.
"""
import math

REF = {}
for b in range(256):
    s = -1.0 if (b & 0x80) else 1.0
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 15 and m == 7:
        REF[b] = 448.0                      # NaN sentinel in e4m3fn -> clamped
    elif e == 0:
        REF[b] = s * (m * 2 ** -9)          # subnormal
    else:
        REF[b] = s * ((1.0 + m / 8.0) * 2 ** (e - 7))


def kernel_e4m3(b):
    """Python mirror of the CUDA e4m3() in sm70_nvfp4_prefill_twopass.py."""
    s = -1.0 if (b & 0x80) else 1.0
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 15 and m == 7:
        return 448.0
    if e == 0:
        return s * (m * 0.001953125)
    return s * ((1.0 + m * 0.125) * math.exp2(e - 7))


bad = []
for b in range(256):
    got, want = kernel_e4m3(b), REF[b]
    if got != want and not (abs(got) == abs(want) == 448.0):
        bad.append((b, got, want))

print(f"e4m3 解码：256 个字节值中 {len(bad)} 个与参考式不符")
for b, g, w in bad[:10]:
    print(f"  byte 0x{b:02x}: 得到 {g!r} 期望 {w!r}")

# the sentinel must be reachable and finite, or the bridge silently mis-scales
sent = [b for b in range(256) if (b >> 3) & 0xF == 15 and b & 7 == 7]
print(f"  448 哨兵字节: {[hex(x) for x in sent]} -> {[kernel_e4m3(x) for x in sent]}")

# subnormals are the easy thing to get wrong in a hand-rolled decode
sub = [b for b in range(256) if ((b >> 3) & 0xF) == 0]
worst_sub = max(abs(kernel_e4m3(b) - REF[b]) for b in sub)
print(f"  次正规 {len(sub)} 个值，最大偏差 {worst_sub:.3e}")

# and confirm the shipped source still matches this mirror, so the test cannot
# silently drift from the kernel it claims to guard
import re
from pathlib import Path

src = Path("/data/src/vllm-v2/vllm/v1/attention/ops/sm70_nvfp4_prefill_twopass.py").read_text()
body = re.search(r"__device__ __forceinline__ float e4m3\(unsigned char b\) \{(.*?)\n\}",
                 src, re.S)
assert body, "e4m3() not found in the kernel source - this test no longer guards it"
code = body.group(1)
for token in ("448.f", "0.001953125f", "0.125f", "exp2f", "(e - 7)"):
    assert token in code, f"e4m3() no longer contains {token!r}; update this test"
print("  源码一致性：e4m3() 的分支常量与本测试一致")
print("VERDICT:", "PASS" if not bad else "FAIL")
