#!/usr/bin/env python3
"""Bit-exactness gate for candidate I (VLLM_SM70_QSA_PLAN_PACK).

The fused kernel must reproduce, element for element,

    idx_p4 = torch.cat([logical_indices[:, :2050],
                        logical_indices[:, 2051:2052].clamp(max=2050)], dim=1)

for the decode shape (1 row), the chunked-prefill envelope (8192 rows), the
empty batch, and — most importantly — the REAL captured selection buffer from
the serving path (/data/sm70_results/qsa_reader_dump.pt, 128x2052 int32).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/data/src/vllm-v2")

from vllm.models.qwen4_exp.nvidia.ops.qsa_plan_pack import (  # noqa: E402
    qsa_plan_pack,
)

REAL_DUMP = "/data/sm70_results/qsa_reader_dump.pt"
IDX_COLS = 2050
OUT_W = 2051
W = 2052


def torch_chain(logical_indices: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            logical_indices[:, :IDX_COLS],
            logical_indices[:, W - 1 : W].clamp(max=IDX_COLS),
        ],
        dim=1,
    )


def compare(name: str, src: torch.Tensor) -> bool:
    ref = torch_chain(src)
    got = qsa_plan_pack(src)
    same = (
        got.shape == ref.shape
        and got.dtype == ref.dtype
        and bool(torch.equal(got, ref))
    )
    extra = ""
    if not same:
        bad = int((got != ref).sum()) if got.shape == ref.shape else -1
        extra = f"  mismatch={bad}"
    print(f"{'PASS' if same else 'FAIL'} {name:<26} rows={src.shape[0]:<6} "
          f"out={tuple(got.shape)} {got.dtype}{extra}")
    return same


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    ok = True

    # Real captured selection buffer from the serving path.
    dump = torch.load(REAL_DUMP, map_location="cpu", weights_only=False)
    real = dump["logical_indices"]
    assert real.shape == (128, W) and real.dtype == torch.int32, real.shape
    ok &= compare("real serving dump", real.to("cuda"))

    # Decode: one row.
    dec = torch.randint(0, 2048, (1, W), dtype=torch.int32, device="cuda")
    dec[0, W - 1] = 700
    ok &= compare("decode rows=1", dec)

    # Chunked-prefill envelope, including counts that must be clamped.
    pre = torch.randint(0, 2100, (8192, W), dtype=torch.int32, device="cuda")
    pre[:, W - 1] = torch.randint(0, 3000, (8192,), dtype=torch.int32, device="cuda")
    ok &= compare("prefill rows=8192", pre)

    # Empty batch.
    empty = torch.empty((0, W), dtype=torch.int32, device="cuda")
    ok &= compare("empty rows=0", empty)

    # Clamp really bites, and only on the trailing column.
    edge = torch.full((2, W), 7, dtype=torch.int32, device="cuda")
    edge[0, W - 1] = 3000  # > 2050 -> clamps
    edge[1, W - 1] = 2050  # exactly at capacity -> unchanged
    got = qsa_plan_pack(edge)
    bit = (
        int(got[0, OUT_W - 1]) == 2050
        and int(got[1, OUT_W - 1]) == 2050
        and bool(torch.equal(got[:, :IDX_COLS], edge[:, :IDX_COLS]))
    )
    print(f"{'PASS' if bit else 'FAIL'} {'count clamp bites only'}   "
          f"counts={int(got[0, OUT_W - 1])},{int(got[1, OUT_W - 1])}")
    ok &= bit

    print("\nVERDICT:", "BIT-EXACT-OK" if ok else "BIT-EXACT-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
