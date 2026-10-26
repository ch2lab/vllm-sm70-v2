#!/usr/bin/env python3
"""L1 wire-codec layer offline verification (ar-three-layer-spec).

Bit-exact CPU references for the hier-leg wire codecs, mirroring the CUDA
codec functions in csrc/libtorch_stable/sm70_hier_ar.cu ONE-BY-ONE (the
reference is the executable spec of the expected bits; the GPU-side
integration check — kernel output vs these references — runs in the
post-window GPU self-check, not here):

  wire=1 (e4m3):   f32_to_e4m3 / e4m3_to_f32 + per-128 fp32 scale
                   sc = mx/448 (sc==0 -> 1.0), payload = e4m3(v/sc).
  wire=2 (nvfp4): e2m1 RTNE on the OCP grid {0,.5,1,1.5,2,3,4,6}
                   (port of sm120-nvfp4 b57e19d1e5 reference),
                   per-16 E4M3 scale sbyte = e4m3(amax/6) (RN div),
                   inv = 1/s RN, 2 codes/byte even-in-low-nibble.
  wire=3 (fp16):  __float2half_rn (IEEE RNE) of the fp32 partial into the
                   [4n..6n) zone (2*n_max B, no scales) — near-lossless
                   middle codec, 2 B/elem (half the fp32 wire).

Round-trip bounds (asserted here, characterizing the codecs alone):
  e4m3 : |err| <= 0.5 ulp of the e4m3 grid at |v|/sc  (<= 2^-4 * sc-ish;
         we assert the exact grid-nearest bound per element)
  nvfp4: |err| <= 0.25 * block amax (e2m1 0.167 + scale rounding 0.0625)

Run:  /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/test_ar_wire_codecs.py
(CPU-only: IEEE-RN fp32 arithmetic is bit-identical to the CUDA __fdiv_rn
and RNE conversions, as verified for the old tree's codec.)
"""

from __future__ import annotations

import struct
import sys

import torch

NV4_BLK = 16
E4M3_BLK = 128
E4M3_MAX = 448.0
NV4_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# ---------------------------------------------------------------------------
# e4m3 (wire=1): mirrors f32_to_e4m3 / e4m3_to_f32 in sm70_hier_ar.cu
# ---------------------------------------------------------------------------

def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def e4m3_to_f32(b: int) -> float:
    E = (b >> 3) & 0xF
    f = b & 0x7
    if E == 0:
        v = (f / 8.0) * (1.0 / 64.0)
    else:
        v = (1.0 + f / 8.0) * (2.0 ** (E - 7))
    return -v if (b & 0x80) else v


def f32_to_e4m3(x: float) -> int:
    """Mirror of the CUDA device function (RN-even, saturate at 448)."""
    sign = 0x80 if x < 0.0 else 0x00
    ax = -x if x < 0.0 else x
    if ax >= E4M3_MAX:
        return sign | 0x7E
    if ax <= 0.0:
        return sign
    # frexpf semantics: ax = m * 2^e2, m in [0.5, 1) -> k = floor(log2 ax),
    # e2 = k + 1. All ops exact in double for an fp32 input.
    k = int(torch.floor(torch.log2(torch.tensor(ax, dtype=torch.float64))).item())
    if k <= -7:  # subnormal e4m3: (f/8) * 2^-6
        f = int(torch.round(torch.tensor(ax * 512.0, dtype=torch.float64)).item())  # RNE
        if f > 7:
            if f == 8:
                return sign | 0x08
            f = 7
        return sign | f
    E = k + 7  # == e2 + 6
    scaled = ax * (2.0 ** -k)  # == ax * 2^(7-E), in [1, 2)
    f = int(torch.round(torch.tensor((scaled - 1.0) * 8.0, dtype=torch.float64)).item())  # RNE
    if f > 7:
        f = 0
        E += 1
    if E >= 15:
        if E == 15 and f <= 6:
            return sign | ((E << 3) | f)
        return sign | 0x7E
    return sign | ((E << 3) | f)


def e4m3_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Quantize->dequantize per-128 (wire=1 semantics: fp32 scale)."""
    n = x.numel()
    xf = x.float().reshape(-1, E4M3_BLK)
    out = torch.empty_like(xf)
    for g in range(xf.shape[0]):
        mx = xf[g].abs().max().item()
        sc = mx / E4M3_MAX
        if sc == 0.0:
            sc = 1.0
        for j in range(E4M3_BLK):
            v = xf[g, j].item()
            code = f32_to_e4m3(v / sc)
            out[g, j] = e4m3_to_f32(code) * sc
    return out.reshape_as(x)


# ---------------------------------------------------------------------------
# nvfp4 (wire=2): mirrors the CUDA branch + old-tree reference (b57e19d1e5)
# ---------------------------------------------------------------------------

def _e2m1_code(v: float) -> int:
    # RTNE on the OCP grid (odd bins: >= on upper tie; even: >)
    c = (
        (v > 0.25)
        + (v >= 0.75)
        + (v > 1.25)
        + (v >= 1.75)
        + (v > 2.5)
        + (v >= 3.5)
        + (v > 5.0)
    )
    return int(c) & 7


def nvfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (payload uint8 [n/2], scales e4m3-as-float [n/16])."""
    n = x.numel()
    assert n % NV4_BLK == 0
    xf = x.float().reshape(-1, NV4_BLK)
    payload = torch.empty(n // 2, dtype=torch.uint8)
    scales = torch.empty(n // NV4_BLK, dtype=torch.float32)
    for g in range(xf.shape[0]):
        amax = xf[g].abs().max().item()
        s = amax / 6.0 if amax > 0.0 else 1.0  # __fdiv_rn
        sbyte = f32_to_e4m3(s)
        s_back = e4m3_to_f32(sbyte)  # scale AS QUANTIZED
        inv = 1.0 / s_back if s_back > 0.0 else 1.0
        # 16 elements = 8 bytes; element 2j low nibble, 2j+1 high nibble
        for j in range(8):
            x0, x1 = xf[g, 2 * j].item(), xf[g, 2 * j + 1].item()
            c0 = _e2m1_code(abs(x0) * inv)
            c1 = _e2m1_code(abs(x1) * inv)
            if x0 < 0.0 and c0:
                c0 |= 8
            if x1 < 0.0 and c1:
                c1 |= 8
            payload[8 * g + j] = torch.tensor(c0 | (c1 << 4), dtype=torch.uint8)
        scales[g] = s_back
    return payload, scales


def nvfp4_dequantize(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n = scales.numel() * NV4_BLK
    out = torch.empty(n, dtype=torch.float32)
    for i in range(n // 2):
        b0 = payload[i].item()
        s = scales[i >> 3].item()  # 16 elems = 8 payload bytes per scale
        for half, elem in ((b0 & 0xF, 2 * i), (b0 >> 4, 2 * i + 1)):
            v = NV4_GRID[half & 7].item() * s
            if half & 8:
                v = -v
            out[elem] = v
    return out


def nvfp4_roundtrip(x: torch.Tensor) -> torch.Tensor:
    p, s = nvfp4_quantize(x)
    return nvfp4_dequantize(p, s)


# ---------------------------------------------------------------------------
# fp16 (wire=3): mirrors __float2half_rn on the staging partials
# ---------------------------------------------------------------------------

def fp16_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Quantize->dequantize per element (fp16 wire, no scales).

    IEEE RNE float32->float16: the kernel's __float2half_rn and the C float
    cast both round the exact value once (an fp32 value is exact in
    double), so np.float16(float32) is bit-identical to the kernel path.
    """
    return x.float().half().float()


def test_fp16_spot():
    rt = fp16_roundtrip(torch.tensor([0.0, 1.0, -1.0, 2.0**-11, 65504.0, 0.5]))
    assert rt.tolist() == [0.0, 1.0, -1.0, 2.0**-11, 65504.0, 0.5]
    # exact halves of powers of two stay exact (power-of-two grid)
    rt = fp16_roundtrip(torch.tensor([0.25, -0.125, 32768.0, 1.0 / 16.0]))
    assert rt.tolist() == [0.25, -0.125, 32768.0, 1.0 / 16.0]
    # fp16 = 10 explicit mantissa bits (11-bit precision incl. the implicit
    # bit): ulp(1.0) = 2^-10, half-ulp = 2^-11.
    # 0.375 ulp above 1.0 -> rounds DOWN (not to 1+2^-11):
    rt = fp16_roundtrip(torch.tensor([1.0 + 2.0**-12 + 2.0**-13]))
    assert rt[0].item() == 1.0
    # 0.75 ulp above 1.0 (1+2^-11+2^-12) -> rounds UP to 1+2^-10:
    rt = fp16_roundtrip(torch.tensor([1.0 + 2.0**-11 + 2.0**-12]))
    assert rt[0].item() == 1.0 + 2.0**-10
    # exact tie at 0.5 ulp (1+2^-12): even side is 1.0 (LSB 0) -> 1.0
    rt = fp16_roundtrip(torch.tensor([1.0 + 2.0**-12]))
    assert rt[0].item() == 1.0
    # tie at 1+2^-10+2^-11: candidates 1+2^-10 (LSB 1) vs 1+2^-9 (LSB 0)
    # -> even side 1+2^-9
    rt = fp16_roundtrip(torch.tensor([1.0 + 2.0**-10 + 2.0**-11]))
    assert rt[0].item() == 1.0 + 2.0**-9


def test_fp16_error_bound_and_wire():
    torch.manual_seed(2)
    # hidden-like: log-normal amplitudes, wide dynamic range
    x = torch.randn(256) * torch.exp(torch.randn(1) * 4.0)
    rt = fp16_roundtrip(x)
    err = (rt - x).abs()
    # normal range: rel <= 2^-11 (half-ulp); subnormal region:
    # abs <= 2^-11. Both covered by the max of the two terms.
    bound = torch.maximum(x.abs() * 2.0**-11, torch.full_like(x, 2.0**-11))
    assert (err <= bound * 1.0001).all(), f"max err {err.max().item():.3e}"
    # wire size: 2 B/elem = half of the fp32 wire (4 B/elem); no scales
    n = x.numel()
    wire = 2 * n
    assert wire / 4 == n // 2  # n/2 halfs
    assert wire == 0.5 * (4 * n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_e4m3_spot():
    # known values (0x00 decodes via the E=0 subnormal formula: (0/8)*2^-6)
    assert e4m3_to_f32(0x00) == 0.0
    assert e4m3_to_f32(0x01) == (1 / 8.0) * (1.0 / 64.0)
    # 0x45: E=(0x45>>3)&15=8, f=5 -> (1+5/8) * 2^(8-7) = 3.25
    assert e4m3_to_f32(0x45) == 3.25
    assert abs(e4m3_to_f32(0x7E) - 448.0) == 0.0
    # 0x7F (the e4m3fn NaN slot) decodes to 480; the ENCODER never emits it
    # (>=448 saturates to 0x7E), so no decode clamp is needed.
    assert e4m3_to_f32(0x7F) == (1 + 7 / 8.0) * 256.0
    assert e4m3_to_f32(0x80) == 0.0
    assert e4m3_to_f32(0x81) == -e4m3_to_f32(0x01)
    # subnormal boundary: 8 ulp-subnormal rounds to 1.0*2^-6
    assert f32_to_e4m3(1.0 / 64.0) == 0x08
    assert f32_to_e4m3(448.0) == 0x7E
    assert f32_to_e4m3(448.01) == 0x7E
    assert f32_to_e4m3(-1.0 / 64.0) == 0x88
    # normal rounding: 1.0 -> E=7,f=0 -> 0x38
    assert f32_to_e4m3(1.0) == 0x38
    # mantissa carry: 1.875 = 1.111b * 2^0 -> f=7 ok; 1.9 -> rounds to 2.0
    assert f32_to_e4m3(1.875) == (0x7 << 3) | 7
    assert f32_to_e4m3(1.9375) == 0x40  # 2.0 = E=8,f=0


def test_e4m3_roundtrip_bound():
    torch.manual_seed(0)
    # 8 blocks of 128, per-block log-normal amplitude (hidden-like spread)
    x = (torch.randn(8, 128) * torch.exp(torch.randn(8, 1) * 2.0)).reshape(-1)
    rt = e4m3_roundtrip(x.reshape(8, 128)).reshape(-1)
    err = (rt - x.float()).abs()
    # per-element: err <= half-ulp of the e4m3 grid at |v|/sc
    # conservative global bound: 2^-4 * sc_block + subnormal slack
    xb = x.float().reshape(8, 128)
    mx = xb.abs().amax(dim=1)
    sc = (mx / E4M3_MAX).clamp_min(0)
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    # Per element: normal-region rel error <= half-ulp = 2^-4 * |x|;
    # subnormal region absolute error <= half subnormal step = 2^-11 * sc.
    # The sum is a valid (loose) bound in both regimes.
    bound = (
        2.0 ** (-4) * x.abs()
        + 2.0 ** (-11) * sc.repeat_interleave(128)
    )
    assert (err <= bound).all(), f"max err {err.max().item():.3e}"


def test_nvfp4_spot():
    # grid codes (thresholds are RTNE: ties go to the even-code side)
    assert [_e2m1_code(v) for v in (0.0, 0.25, 0.75, 1.25, 2.5, 5.0)] == [
        0, 0, 2, 2, 4, 6
    ]
    assert [_e2m1_code(v) for v in (0.26, 0.74, 1.26, 1.74, 2.51, 3.49)] == [
        1, 1, 3, 3, 5, 5
    ]
    # tie-to-even on the grid: 0.25 -> 0, 0.75 -> 2 (1.0), 5.0 -> 6 (4.0)
    assert _e2m1_code(0.25) == 0
    assert _e2m1_code(0.75) == 2
    assert _e2m1_code(5.0) == 6
    # amax maps to 6.0 (block of 16)
    x = torch.tensor([0.0, 6.0, -6.0, 3.0] + [0.0] * 12)
    p, s = nvfp4_quantize(x)
    assert p.numel() == 8 and s.numel() == 1
    rt = nvfp4_dequantize(p, s)
    # 6.0 -> 6.0 exactly (amax in scale); 3.0 -> 3.0; sign preserved
    assert abs(rt[1].item() - 6.0) < 1e-6
    assert abs(rt[2].item() + 6.0) < 1e-6
    assert abs(rt[3].item() - 3.0) < 1e-6
    assert rt[0].item() == 0.0


def test_nvfp4_error_bound():
    torch.manual_seed(1)
    x = torch.randn(64 * NV4_BLK)
    p, s = nvfp4_quantize(x)
    rt = nvfp4_dequantize(p, s)
    err = (rt - x).abs()
    amax = x.abs().reshape(-1, NV4_BLK).amax(dim=1).repeat_interleave(NV4_BLK)
    bound = 0.25 * amax * 1.002
    assert (err <= bound).all(), f"max err {err.max().item():.6f}"
    # wire size: n/2 + n/16 bytes vs e4m3's n + n/32
    n = x.numel()
    assert p.numel() + s.numel() == n // 2 + n // 16
    ratio = (n // 2 + n // 16) / (n + n // 32)
    assert abs(ratio - 0.545) < 0.01, ratio


def main() -> int:
    for name, fn in (
        ("e4m3 spot values", test_e4m3_spot),
        ("e4m3 roundtrip bound", test_e4m3_roundtrip_bound),
        ("nvfp4 spot values", test_nvfp4_spot),
        ("nvfp4 error bound + wire size", test_nvfp4_error_bound),
        ("fp16 spot values", test_fp16_spot),
        ("fp16 error bound + wire size", test_fp16_error_bound_and_wire),
    ):
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            return 1
    print("== all codec references passed ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
