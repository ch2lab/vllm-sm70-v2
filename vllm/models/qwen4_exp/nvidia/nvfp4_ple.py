# SPDX-License-Identifier: Apache-2.0
"""Pure-torch NVFP4 (E2M1 + per-16 E4M3) quantize/dequant reference for PLE.

The PLE table is stored as FP8 today (one byte/element, ~51.2 GB pinned at
20M x 2560). NVFP4 stores each element as a 4-bit E2M1 code plus a per-16-element
E4M3 group scale, halving the table to ~29 GB pinned. This module is the
correctness reference that the Triton PLE lookup kernel must match: it provides
the E2M1 code table, the E4M3 scale table, a quantize (fp16 -> packed codes +
group scales), and a decode (packed codes + scales -> fp16).

The E2M1 table mirrors ``nvfp4_e2m1_to_half`` in
``csrc/libtorch_stable/quantization/marlin/sm70_nvfp4_dequant.cu``; the E4M3
table is torch's own ``float8_e4m3fn`` cast (the same convention the FP8 PLE LUT
already uses). The quantize/dequant are self-consistent by construction: both
side-step the same two tables and the same low-even / high-odd packing.
"""

import torch

_LUT_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def nvfp4_e2m1_lut(device) -> torch.Tensor:
    """16-entry E2M1 code -> value table (fp16).

    Mirrors the C++ dequant decode: E2=(code>>1)&3, M1=code&1, sign=code>>3.
    """
    key = ("e2m1", str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        base = (0.0, 1.0, 2.0, 4.0)  # base[e2] for e2 in 1..3
        # Create on CPU first (avoids meta-tensor copy issues with IPC loader),
        # then move to target device.
        lut = torch.empty(16, dtype=torch.float16, device="cpu")
        for code in range(16):
            e2 = (code >> 1) & 3
            m1 = code & 1
            v = (0.5 if m1 else 0.0) if e2 == 0 else base[e2] * (1.5 if m1 else 1.0)
            lut[code] = -v if (code & 0x8) else v
        if device.type != "meta":
            lut = lut.to(device)
        _LUT_CACHE[key] = lut
    return lut


def nvfp4_e4m3_lut(device) -> torch.Tensor:
    """256-entry E4M3 code -> value table (fp16), torch's own float8_e4m3fn cast."""
    key = ("e4m3", str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        codes = torch.arange(256, dtype=torch.uint8, device="cpu")
        lut = (
            codes.view(torch.float8_e4m3fn)
            .to(torch.float16)
            .contiguous()
        )
        if device.type != "meta":
            lut = lut.to(device)
        _LUT_CACHE[key] = lut
    return lut


def nvfp4_quantize(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``y`` (fp16, ``... x dim`` with ``dim % 16 == 0``) to NVFP4.

    Returns ``(codes, sf)``: ``codes`` is uint8 ``( ... x dim//2 )`` (low nibble =
    even k, high nibble = odd k) and ``sf`` is uint8 ``( ... x dim//16 )`` (one
    E4M3 group scale per 16 elements). The decode satisfies
    ``nvfp4_decode(codes, sf) ~= y``.
    """
    y = y.contiguous()
    device = y.device
    e2 = nvfp4_e2m1_lut(device).float()
    e4 = nvfp4_e4m3_lut(device).float()
    orig = y.shape
    dim = orig[-1]
    ngroups = dim // 16
    rows = y.numel() // dim
    g = y.reshape(rows, ngroups, 16).float()

    amax = g.abs().amax(dim=-1)
    sf_target = amax / 6.0
    # Group scales are non-negative, so only the finite E4M3 codes 0..0x7E apply.
    e4_pos = e4[: 0x7F]  # 127 values: 0..448 (excludes 0x7F = NaN)
    sf_code = (e4_pos.view(1, 1, -1) - sf_target[..., None]).abs().argmin(dim=-1)
    sf_code = sf_code.to(torch.uint8)
    sf_val = e4_pos[sf_code.to(torch.long)]
    q = g / sf_val.clamp_min(torch.finfo(torch.float32).tiny)[..., None]

    code = (e2.view(1, 1, -1) - q[..., None]).abs().argmin(dim=-1)
    cp = code.view(rows, ngroups, 8, 2).to(torch.uint8)
    packed = (cp[..., 0] | (cp[..., 1] << 4)).reshape(rows, ngroups * 8)
    return packed.view(*orig[:-1], dim // 2), sf_code.view(*orig[:-1], ngroups)


def nvfp4_decode(codes: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Decode NVFP4 ``codes`` (uint8, ``... x dim//2``) + ``sf`` (uint8,
    ``... x dim//16``) back to fp16 ``... x dim`` (no global scale applied)."""
    device = codes.device
    e2 = nvfp4_e2m1_lut(device)
    e4 = nvfp4_e4m3_lut(device)
    orig = codes.shape
    dim = orig[-1] * 2
    ngroups = dim // 16
    rows = codes.numel() // (dim // 2)
    c = codes.view(rows, ngroups, 8).to(torch.int32)
    lo = (c & 0xF)
    hi = (c >> 4) & 0xF
    code = torch.stack([lo, hi], dim=-1).view(rows, ngroups, 16)
    e2v = e2[code.to(torch.long)]
    sfv = e4[sf.view(rows, ngroups).to(torch.long)]
    return (e2v * sfv.unsqueeze(-1)).view(*orig[:-1], dim)


def nvfp4_from_fp8(fp8_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert FP8 PLE codes to NVFP4 ``(codes, sf)``.

    The FP8 PLE stores ``value = fp8_code * weight_scale`` (per-tensor
    ``weight_scale``). To reconstruct the same value as NVFP4 we requantize the
    fp8 *codes* -- which already sit in the E4M3 range [-448, 448], so per-16
    group scales stay in the E4M3 normal range -- and keep ``weight_scale`` as
    the NVFP4 global scale applied at decode (as with the FP8 path).
    """
    return nvfp4_quantize(fp8_codes.to(torch.float16))
