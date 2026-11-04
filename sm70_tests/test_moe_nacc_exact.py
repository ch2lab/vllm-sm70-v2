#!/usr/bin/env python3
"""Quantify what NAcc=2 costs in accuracy, on the production decode shapes.

Two independent fp32 accumulator chains change the association of the K
reduction, so this change CANNOT be validated by bit-identity (unlike the
combine_decode hoist, where the reduction order was untouched).  The test is
therefore: same inputs, two launcher arms, measure the difference and check it
is at fp32-accumulation noise level.

Each arm runs in its own process because VLLM_SM70_MOE_QPN2_NACC is read once
into a static.

    bash sm70_tests/test_moe_nacc_exact.sh   # dumps + compares
"""
from __future__ import annotations

import hashlib
import sys

import torch


def load(path: str) -> torch.Tensor:
    raw = open(path, "rb").read()
    return torch.frombuffer(bytearray(raw), dtype=torch.int16).view(
        torch.float16).float()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    assert a.shape == b.shape, (a.shape, b.shape)
    ha = hashlib.sha256(a.half().cpu().numpy().tobytes()).hexdigest()[:16]
    hb = hashlib.sha256(b.half().cpu().numpy().tobytes()).hexdigest()[:16]
    d = (a - b).abs()
    denom = a.abs().amax().clamp(min=1e-6)
    n_bit = int((a.half().view(torch.int16) != b.half().view(torch.int16)).sum())
    print(f"  elements={a.numel():,}  sha {ha} vs {hb}")
    print(f"  max|d|={float(d.max()):.3e}  max rel={float(d.max()/denom):.3e}  "
          f"mean|d|={float(d.mean()):.3e}")
    print(f"  differing fp16 bit patterns: {n_bit:,} "
          f"({100.0*n_bit/a.numel():.4f}%)  "
          f"ulp-of-max={float(d.max()/ (denom/2048.0)):.2f}")
    ok = float(d.max() / denom) < 1e-3
    print("VERDICT:", "within fp32-reassociation noise (<1e-3 rel)" if ok
          else "*** TOO LARGE ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
