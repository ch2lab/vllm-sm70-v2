"""Fused ``out = sigmoid(gate) * out`` for the shared-expert gate.

``Qwen2MoeMLP`` (aliased as ``Qwen3NextMLP``, hence every layer of the served
Qwen3.8 model) runs the shared-expert output gate as two ATen launches:
``F.sigmoid(self.expert_gate(x)[0])`` over ``[tokens, 1]`` and the broadcast
multiply over ``[tokens, hidden]``. At decode that pair costs ~0.61 ms/step
(48 layers) in this service, and both kernels are launch-bound, not
bandwidth-bound.

The fused kernel is numerically identical to the two-step form: the sigmoid is
rounded to the accumulate dtype before the multiply, exactly like the separate
``aten::sigmoid`` tensor followed by ``aten::mul``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_mul_kernel(
    out_ptr,
    gate_ptr,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Round the gate through the output dtype first: matches
    # ``F.sigmoid(gate)`` materialising fp16 before the multiply.
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g).to(tl.float16).to(tl.float32)
    base = row * n_cols
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(out_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + base + cols,
            (x * g).to(tl.float16),
            mask=mask,
        )


def fused_sigmoid_mul(
    out: torch.Tensor,
    gate: torch.Tensor,
    *,
    block: int = 1024,
) -> torch.Tensor:
    """``out * sigmoid(gate)`` in one launch; falls back to the ATen pair.

    ``out`` is ``[tokens, hidden]`` fp16 and is updated in place; ``gate`` is
    ``[tokens]`` or ``[tokens, 1]``.
    """
    if (
        out.dim() == 2
        and out.is_cuda
        and out.dtype == torch.float16
        and out.is_contiguous()
        and gate.is_cuda
        and gate.dtype == torch.float16
        and gate.is_contiguous()
        and gate.numel() == out.shape[0]
    ):
        _sigmoid_mul_kernel[(out.shape[0],)](
            out,
            gate,
            out.shape[1],
            BLOCK=block,
            num_warps=4,
        )
        return out
    return torch.sigmoid(gate) * out
