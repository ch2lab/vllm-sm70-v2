# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (V100) Triton top-k softmax routing kernel.

The CUDA `topkGating` kernel (vllm::_moe_C) handles a row with one active
warp doing K sequential butterfly-argmax rounds; at E=512 / K=10 / B=1 that
measured 13.2us/call graphed (tiny-1CTA control floor is 1.3us). This Triton
kernel does the same softmax-topk-renormalize per row in a single program
(512 values in registers, K sequential argmax reductions; tl.argmax breaks
ties by smallest index, verified against the CUDA kernel), and measures
~2.2x faster graphed (6.0us vs 13.2us at E=512/K=10/B=1).

Knob: VLLM_SM70_TRITON_TOPK=1 routes FusedTopKRouter's softmax scoring
through this kernel (see vllm/model_executor/layers/fused_moe/router/
fused_topk_router.py). Semantics match ops.topk_softmax:
  - softmax over E logits (max-shifted, translation-invariant),
  - top-K in descending value order, ties broken by smallest expert index,
  - renormalize: weights divided by the sum of the selected softmax values
    (sum<=0 -> unnormalized, same guard as the CUDA kernel),
  - degenerate rows (all -inf / NaN-free underflow): weights 0, ids 0..K-1.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# Fast-launch cache: the standard triton JITFunction call costs ~30us of
# CPU per launch (vs ~13us for the CUDA op); at 48 calls/decode step that
# 30us x 48 = 1.5ms/step of Python time regresses the serve TPOT even
# though the kernel itself is 2.2x faster. First call per key goes through
# the normal path (compile + cache lookup), subsequent calls invoke the
# compiled kernel's launcher directly.
_fast_cache: dict = {}
_fast_disabled: set = set()


@triton.jit
def _sm70_topk_softmax_kernel(
    gate_ptr,
    w_ptr,
    id_ptr,
    te_ptr,
    pad_ptr,
    stride_gb,
    stride_wb,
    stride_ib,
    stride_tb,
    renormalize: tl.constexpr,
    HAS_PAD: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    K2: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0)
    num_rows = tl.num_programs(0)
    if HAS_PAD:
        is_pad = tl.load(pad_ptr + row)
    else:
        is_pad = 0
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    x = tl.load(
        gate_ptr + row * stride_gb + offs, mask=mask, other=float("-inf")
    ).to(tl.float32)
    # Max-shift for stability; flooring the shift at 0 keeps all-negative rows
    # from underflowing differently (softmax is translation-invariant, so
    # weights are identical for any finite row).
    mx = tl.maximum(tl.max(x, axis=0), 0.0)
    p = tl.exp(x - mx)
    s = tl.sum(p, axis=0)
    p = p / tl.where(s > 0, s, 1.0)

    # K2 = next power of two >= K (tl.arange requires it); mask trims to K.
    idx_k = tl.arange(0, K2)
    mask_k = idx_k < K
    ws = tl.zeros((K2,), tl.float32)
    sel_sum = 0.0
    for ki in tl.static_range(K):
        # The row max IS the selected value (j is an argmax position).
        j = tl.argmax(p, axis=0)
        v = tl.max(p, axis=0)
        sel_sum += v
        ws = tl.where(idx_k == ki, v, ws)
        # CUDA contract: padded rows get indices -1 (weights still computed);
        # token_expert_indices always holds the flattened source-row index
        # k * num_rows + row (grouped-GEMM gather order).
        tl.store(id_ptr + row * stride_ib + ki,
                 tl.where(is_pad, -1, j).to(tl.int32))
        tl.store(te_ptr + row * stride_tb + ki, (ki * num_rows + row).to(tl.int32))
        p = tl.where(offs == j, float("-inf"), p)
    if renormalize:
        ws = ws / tl.where(sel_sum > 0, sel_sum, 1.0)
    tl.store(w_ptr + row * stride_wb + idx_k, ws, mask=mask_k)


def sm70_triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    is_padding: torch.Tensor | None = None,
) -> None:
    """In-place top-k softmax into the preallocated outputs (same contract as
    torch.ops._moe_C.topk_softmax). One Triton program per row. Padded rows
    (is_padding) get indices -1 with weights computed as usual, matching the
    CUDA kernel."""
    assert gating_output.stride(1) == 1
    assert topk_weights.stride(1) == 1
    assert topk_indices.stride(1) == 1
    assert token_expert_indices.stride(1) == 1
    b, e = gating_output.shape
    if b == 0:
        return
    k = topk_indices.shape[1]
    assert e <= 1024
    block_e = max(triton.next_power_of_2(e), k * 2)
    k2 = triton.next_power_of_2(k)
    has_pad = is_padding is not None
    pad = is_padding if is_padding is not None else topk_indices
    ren = bool(renormalize)
    strides = (gating_output.stride(0), topk_weights.stride(0),
               topk_indices.stride(0), token_expert_indices.stride(0))

    # Fast path: direct compiled-kernel launch (skips JIT dispatch).
    key = (e, k, b, gating_output.dtype, ren, has_pad)
    fast = _fast_cache.get(key)
    if fast is not None:
        ck, dev_idx = fast
        # Raw C binding: ~0.3us vs ~6us for torch.cuda.current_stream().
        stream = torch._C._cuda_getCurrentRawStream(dev_idx)
        ck.run(b, 1, 1, stream, ck.function, ck.packed_metadata, None,
               None, None, gating_output, topk_weights, topk_indices,
               token_expert_indices, pad, *strides)
        return
    if key in _fast_disabled:
        pass
    else:
        # Normal path first (compiles if needed), then grab the compiled
        # kernel that was just produced/looked-up for future fast calls.
        _sm70_topk_softmax_kernel[(b,)](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            pad,
            *strides,
            renormalize=ren,
            HAS_PAD=has_pad,
            E=e,
            K=k,
            K2=k2,
            BLOCK_E=block_e,
            num_warps=4,
        )
        try:
            kc = _sm70_topk_softmax_kernel.device_caches[
                torch.cuda.current_device()][0]
            _fast_cache[key] = (next(reversed(kc.values())),
                                gating_output.device.index)
        except Exception:
            _fast_disabled.add(key)
        return
    _sm70_topk_softmax_kernel[(b,)](
        gating_output,
        topk_weights,
        topk_indices,
        token_expert_indices,
        pad,
        *strides,
        renormalize=ren,
        HAS_PAD=has_pad,
        E=e,
        K=k,
        K2=k2,
        BLOCK_E=block_e,
        num_warps=4,
    )
