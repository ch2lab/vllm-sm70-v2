# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate H shadow: single-launch step-invariant QSA row positions.

``Qwen4ExpQSAFlashAttentionImpl.forward_qsa`` recomputes, for EVERY QSA layer
(12 per step), the Flash-V100 page4 planner's absolute row positions:

    local = arange(n) - qsl[token_to_req]
    qp    = (seq_lens[token_to_req] - (qsl[1:] - qsl[:-1])[token_to_req]
             + local).to(int64)

That is 8-9 tiny index kernels per layer, all pure integer bookkeeping over
metadata that is already fully materialized for the step. The measured
in-graph launch cost on this box is 4.6 us (``VLLM_SM70_LAUNCH_PROBE``), so
those ~90 launches/step are pure overhead.

Wiring: gated by ``VLLM_SM70_QSA_POS_FUSE`` (default 0 = today). With 0 this
module is never imported and the existing torch chain runs unchanged.

Deliberately STATELESS. A Python-level cache keyed on the metadata tensors
would be a correctness bug here: those are static cudagraph buffers whose
pointers stay stable across steps while their values change, and under the
breakable cudagraph the eager pieces still execute Python at replay while
the pure-replay pieces do not - so "compute once, reuse for the later
layers" does not hold. One fused kernel per layer keeps per-step semantics
with no cross-step state.

Numerics: the arithmetic is pure int64 integer math on the same operands and
the same formula, so the output is bit-identical (asserted by
``sm70_tests/test_qsa_pos_fuse.py`` against the torch chain).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _qsa_pos_kernel(
        token_to_req_ptr,
        qsl_ptr,
        seq_lens_ptr,
        out_ptr,
        n_tokens,
        BLOCK: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_tokens
        ridx = tl.load(token_to_req_ptr + offs, mask=m, other=0).to(tl.int64)
        # qsl is [num_reqs + 1]; token_to_req[i] < num_reqs so ridx + 1 is
        # always in bounds.
        q0 = tl.load(qsl_ptr + ridx, mask=m, other=0).to(tl.int64)
        q1 = tl.load(qsl_ptr + ridx + 1, mask=m, other=0).to(tl.int64)
        s = tl.load(seq_lens_ptr + ridx, mask=m, other=0).to(tl.int64)
        pos = s - (q1 - q0) + (offs.to(tl.int64) - q0)
        tl.store(out_ptr + offs, pos, mask=m)


def qsa_positions_fused(
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_tokens: int,
    *,
    block: int = 256,
) -> torch.Tensor:
    """Absolute per-row positions, identical to the torch chain.

    Args:
        token_to_req: int32 [num_tokens] request index of each token.
        query_start_loc: int32 [num_reqs + 1] cumulative query starts.
        seq_lens: int32 [num_reqs] per-request sequence lengths.
        num_tokens: number of live tokens.

    Returns:
        int64 [num_tokens] ``seq_lens[r] - qlen[r] + (i - qsl[r])``.
    """
    out = torch.empty(num_tokens, dtype=torch.int64, device=seq_lens.device)
    if num_tokens == 0:
        return out
    grid = (triton.cdiv(num_tokens, block),)
    _qsa_pos_kernel[grid](
        token_to_req,
        query_start_loc,
        seq_lens,
        out,
        num_tokens,
        BLOCK=block,
    )
    return out


__all__ = ["qsa_positions_fused"]
