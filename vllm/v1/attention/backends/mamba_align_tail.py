# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText= Copyright contributors to the vLLM project

"""Fused "align" state-block-table fetch for mamba/GDN attention.

``mamba_get_block_table_tensor``'s align branch selects each request's
``1 + num_speculative_blocks`` state-block ids starting at the block that holds
its last token. Written with aten that is five ops per call (subtract,
floor-divide, clamp, broadcast-add, gather) and it runs once per KV-cache group
per *decode step* -- three groups here, so 15 tiny dispatches per step at
~20-25 us of host time each, on the critical path in front of every CUDA-graph
replay. This module does the same arithmetic in one launch.

The index maths matches the aten formulation bit for bit, including the
0-length-request behaviour (both clamp the start index to 0). Column indices at
or past the table width are stored as 0 rather than trapping the device, which
is strictly a superset of the previous behaviour: that case raised a gather
index assertion before, so no existing caller can depend on it.
"""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv

if HAS_TRITON:

    @triton.jit
    def _mamba_align_tail_kernel(
        bt_ptr,  # [num_rows, width] int32 state-block ids, one row per request
        seq_lens_ptr,  # [num_rows] int32 sequence lengths
        out_ptr,  # [num_rows, N_COLS] int32, the fetched tail
        bt_stride_r,
        out_stride_r,
        width,
        num_rows,
        BS: tl.constexpr,  # kv_cache_spec.block_size
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,  # next_pow2(N_COLS)
        BLOCK_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row_ok = rows < num_rows
        # Truncating division on a negative numerator gives 0 here, and the
        # clamp below makes that agree with aten's floor-then-clamp exactly.
        seq_lens = tl.load(seq_lens_ptr + rows, mask=row_ok, other=1).to(tl.int32)
        start = tl.maximum((seq_lens - 1) // BS, 0)

        col_ok = cols < N_COLS
        idx = start[:, None] + cols[None, :]
        load_ok = row_ok[:, None] & col_ok[None, :] & (idx < width) & (idx >= 0)
        vals = tl.load(
            bt_ptr + rows[:, None] * bt_stride_r + idx,
            mask=load_ok,
            other=0,
        )
        tl.store(
            out_ptr + rows[:, None] * out_stride_r + cols[None, :],
            vals,
            mask=row_ok[:, None] & col_ok[None, :],
        )


def mamba_align_tail(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    n_cols: int,
) -> torch.Tensor | None:
    """Return ``[num_rows, n_cols]`` int32, or None if the fused path is
    unavailable (caller then uses the aten formulation)."""
    from vllm import envs

    if not envs.VLLM_SM70_MAMBA_ALIGN_TAIL:
        # Caller falls back to the aten formulation. Import inside the function so this
        # module keeps importing without a configured environment.
        return None
    if not HAS_TRITON or block_size <= 0 or n_cols <= 0:
        return None
    num_rows = seq_lens.shape[0]
    width = block_table.shape[1]
    out = torch.empty(
        (num_rows, n_cols),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    if num_rows == 0:
        return out
    BLOCK_ROWS = 32
    _mamba_align_tail_kernel[(cdiv(num_rows, BLOCK_ROWS),)](
        block_table,
        seq_lens,
        out,
        block_table.stride(0),
        out.stride(0),
        width,
        num_rows,
        BS=block_size,
        N_COLS=n_cols,
        BLOCK_COLS=1 << (n_cols - 1).bit_length(),
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=4,
    )
    return out
