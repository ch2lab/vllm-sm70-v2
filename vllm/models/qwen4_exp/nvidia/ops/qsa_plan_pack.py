# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate I shadow: one-launch repack of the QSA planner's index block.

Both page4 routes hand the C++ planner a fixed [rows, 2051] ABI block (2050
index columns + a count column), while the selector's packed buffer is one
index column wider with the count at the last column. Today that repack is

    idx_p4 = torch.cat([logical_indices[:, :2050],
                        logical_indices[:, 2051:2052].clamp(max=2050)], dim=1)

i.e. two in-graph launches (clamp + cat) for every QSA layer (12/step), plus a
full-width temporary at prefill sizes. This replaces them with ONE launch that
writes the ABI block directly.

Wiring: gated by ``VLLM_SM70_QSA_PLAN_PACK`` (default 0 = today; the module is
only imported when the valve is on). Stateless, and pure integer element
movement, so the result is bit-identical to the torch chain.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _qsa_plan_pack_kernel(
        src_ptr,
        dst_ptr,
        stride_s,
        stride_d,
        total,
        W: tl.constexpr,
        OUT_W: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        row = offs // OUT_W
        col = offs % OUT_W
        # The last destination column carries the source's trailing count
        # column, clamped to the planner's index capacity (OUT_W - 1).
        is_count = col == (OUT_W - 1)
        src_col = tl.where(is_count, W - 1, col)
        v = tl.load(src_ptr + row * stride_s + src_col, mask=m, other=0)
        v = tl.where(is_count, tl.minimum(v, OUT_W - 1), v)
        tl.store(dst_ptr + row * stride_d + col, v, mask=m)


def qsa_plan_pack(
    logical_indices: torch.Tensor,
    out_width: int = 2051,
    *,
    block: int = 1024,
) -> torch.Tensor:
    """Pack the selection buffer into the planner's [rows, out_width] ABI.

    Args:
        logical_indices: int32 [rows, W] packed selection; the trailing column
            (W - 1) is the row's valid-entry count.
        out_width: planner ABI width (2050 index columns + count).

    Returns:
        int32 [rows, out_width] with the count column clamped to out_width - 1.
    """
    rows, width = logical_indices.shape
    out = torch.empty(
        (rows, out_width), dtype=logical_indices.dtype, device=logical_indices.device
    )
    if rows == 0:
        return out
    total = rows * out_width
    grid = (triton.cdiv(total, block),)
    _qsa_plan_pack_kernel[grid](
        logical_indices,
        out,
        logical_indices.stride(0),
        out.stride(0),
        total,
        W=width,
        OUT_W=out_width,
        BLOCK=block,
    )
    return out


__all__ = ["qsa_plan_pack"]
