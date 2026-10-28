# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fused checkpoint-FP16 HyperConnection decode route for SM70.

Ported from 1Cat-vLLM (same file). For the exact Qwen3.8 TP4 single-request
N=1 decode shape, each GatedResidual mix is replaced by:

  - TP4-sharded Triton down projection (this rank's 80 lora rows + 1
    injection row, replicated weight layout read from the checkpoint row
    mapping);
  - a one-shot push allgather of the 88-element packet per GatedResidual
    (custom_all_reduce.cuh sm70_qwen38_hc_*);
  - either the fully fused up/mix/push CUDA kernel (cooperative launch), or
    the sharded Triton up projection + output allgather / gate mix fallbacks.

All FP16 materialization boundaries (GEMV -> FP16 -> SiLU / sigmoid) match
the baseline down-GEMV + SiLU + up-GEMV + gate-mix sequence bit-for-bit, so
the fused route and the standard route are numerically interchangeable.
"""

from __future__ import annotations

import os

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_HC_COUNT = 4
_HC_UP_STREAM4 = os.environ.get("VLLM_SM70_HC_UP_STREAM4", "0") == "1"
_HC_UP_STREAM4_WARPS = int(os.environ.get("VLLM_SM70_HC_UP_STREAM4_WARPS", "8"))
_HC_DIM = 2560
# Columns per program for the up-projection; 1 keeps the one-column kernel.
# 8 makes each lane hold eight consecutive fp16 weights so the read coalesces.
_HC_UP_WIDE = int(os.environ.get("VLLM_SM70_HC_UP_WIDE", "8"))
_HC_RANK = 320
_HC_HIDDEN = _HC_COUNT * _HC_DIM


def _hc_lowrank_of(child) -> object:
    """HC low-rank width; the module exposes `hc_lowrank` (hyperconnection.py),
    older revisions named it `lora_rank` — accept both, else every fusion below
    silently finds no targets (regression observed 2026-10-22)."""
    value = getattr(child, "hc_lowrank", None)
    return value if value is not None else getattr(child, "lora_rank", None)


def _exact_runtime_contract(vllm_config=None) -> bool:
    try:
        config = vllm_config or get_current_vllm_config()
        text_config = config.model_config.hf_text_config
        tp_size = int(config.parallel_config.tensor_parallel_size)
    except (AssertionError, AttributeError, RuntimeError):
        return False

    return bool(
        tp_size == 4
        and config.speculative_config is None
        and int(getattr(text_config, "hidden_size", 0)) == 2560
        and int(getattr(text_config, "num_hidden_layers", 0)) == 48
        and int(getattr(text_config, "num_experts", 0)) == 512
        and int(getattr(text_config, "num_experts_per_tok", 0)) == 10
        and int(getattr(text_config, "moe_intermediate_size", 0)) == 640
        and int(getattr(text_config, "hc_count", 0)) == 4
        and int(getattr(text_config, "hc_lowrank", 0)) == 320
        and int(getattr(text_config, "num_attention_heads", 0)) == 24
        and int(getattr(text_config, "num_key_value_heads", 0)) == 2
        and int(getattr(text_config, "indexer_head_dim", 0)) == 128
        and int(getattr(text_config, "indexer_budget", 0)) == 2048
        and int(getattr(text_config, "indexer_compress_ratio", 0)) == 4
    )


@triton.jit
def _qwen38_hc_down_silu_inject_kernel(
    x_ptr,
    weight_ptr,
    lora_ptr,
    injection_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(
            x_ptr + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        weight = tl.load(
            weight_ptr + row * K + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Preserve the baseline GEMV -> FP16 -> SiLU boundary.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    is_lora = row < RANK_VALUE
    scaled = value / HC_COUNT
    tl.store(lora_ptr + row, scaled * tl.sigmoid(scaled), mask=is_lora)
    tl.store(injection_ptr + row - RANK_VALUE, value, mask=~is_lora)


@triton.jit
def _qwen38_hc_down_local_shard_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    TP_RANK: tl.constexpr,
):
    """Compute this TP rank's 80 low-rank rows and one injection row."""
    row = tl.program_id(0)
    active = row < 81
    checkpoint_row = tl.where(row < 80, TP_RANK * 80 + row, 320 + TP_RANK)
    offsets = tl.arange(0, 256)
    acc = tl.zeros((256,), dtype=tl.float32)
    for block_start in tl.static_range(0, 10240, 256):
        indices = block_start + offsets
        x = tl.load(
            x_ptr + indices,
            mask=active,
            other=0.0,
            eviction_policy="evict_last",
        )
        weight = tl.load(
            weight_ptr + checkpoint_row * 10240 + indices,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Match the replicated projection's FP16 materialization before SiLU.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    scaled = value / 4
    value = tl.where(row < 80, scaled * tl.sigmoid(scaled), value)
    tl.store(output_ptr + row, value, mask=active)
    # Keep the 88-element communication packet aligned to 16 bytes. Padding
    # is canonical zero and is discarded after the rank-ordered gather.
    tl.store(output_ptr + row, 0.0, mask=~active)


@triton.jit
def _qwen38_hc_up_local_gate_kernel(
    lora_ptr,
    weight_ptr,
    gate_ptr,
    TP_RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute the 2560 gate rows owned by this TP rank."""
    hidden = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, 512)
    hidden_mask = hidden < 2560
    k_mask = offsets < 320
    lora = tl.load(
        lora_ptr + offsets,
        mask=k_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    checkpoint_row = TP_RANK * 2560 + hidden
    weight = tl.load(
        weight_ptr + checkpoint_row[:, None] * 320 + offsets[None, :],
        mask=hidden_mask[:, None] & k_mask[None, :],
        other=0.0,
        eviction_policy="evict_first",
    )
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    # The communication kernel applies the original FP16 gate boundary,
    # sigmoid, rank-ordered FP32 FMA, and final FP16 materialization.
    tl.store(gate_ptr + hidden, gate, mask=hidden_mask)


@triton.jit
def _qwen38_hc_up_hidden_shard_kernel(
    lora_ptr,
    weight_ptr,
    branches_ptr,
    out_ptr,
    TP_RANK: tl.constexpr,
):
    """Mix all four branches locally for two of this rank's 640 hidden rows."""
    rows = tl.arange(0, 8)
    hidden = tl.program_id(0) * 2 + rows // 4
    checkpoint_row = (rows % 4) * 2560 + TP_RANK * 640 + hidden
    offsets = tl.arange(0, 512)
    lora = tl.load(lora_ptr + offsets, offsets < 320, 0).to(tl.float32)
    weight = tl.load(
        weight_ptr + checkpoint_row[:, None] * 320 + offsets[None, :],
        offsets[None, :] < 320,
        0,
    )
    # Keep the existing two-K-warp reduction, FP16 gate boundary, and
    # branch-ordered FP32 FMA. Only row ownership changes; weights are neither
    # repacked nor duplicated, and prefill keeps its original layout.
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    gate = gate.to(tl.float16).to(tl.float32).reshape((2, 4))
    branches = tl.load(branches_ptr + checkpoint_row).to(tl.float32).reshape(
        (2, 4)
    )
    result = tl.full((2,), 0, tl.float32)
    for branch in tl.static_range(4):
        index = tl.full((2, 1), branch, tl.int32)
        g = tl.gather(gate, index, 1).reshape((2,))
        x = tl.gather(branches, index, 1).reshape((2,))
        result = tl.fma(tl.sigmoid(g), x, result)
    tl.store(out_ptr + tl.program_id(0) * 2 + tl.arange(0, 2), result / 4)


@triton.jit
def _qwen38_hc_up_gate_mix_row4_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reuse the low-rank input across four bitwise-equivalent output rows."""
    hidden = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    hidden_mask = hidden < HC_DIMENSION
    k_mask = offsets < K
    lora = tl.load(
        lora_ptr + offsets,
        mask=k_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for stream in tl.static_range(HC_COUNT):
        row = stream * HC_DIMENSION + hidden
        weight = tl.load(
            weight_ptr + row[:, None] * K + offsets[None, :],
            mask=hidden_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        # Keep the established FP32 reduction and FP16 gate boundary. Row
        # tiling changes only work assignment and shares the lora read.
        gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
        gate = gate.to(tl.float16).to(tl.float32)
        branch = tl.load(
            x_ptr + stream * HC_DIMENSION + hidden,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        result += tl.sigmoid(gate) * branch
    tl.store(out_ptr + hidden, result / HC_COUNT, mask=hidden_mask)


@triton.jit
def _qwen38_hc_up_gate_mix_stream4_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One hidden unit per program, all HC_COUNT streams in a single tile.

    Same per-row arithmetic as the row4 kernel (identical [1, BLOCK_K] x
    [1, BLOCK_K] product and the same reduction over K, the same fp16 gate
    boundary), but the streams are reduced together instead of four
    dependent reduce chains. The final accumulation keeps the production
    stream order so the result is bit-identical (verified with torch.equal).
    """
    hidden = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    k_mask = offs < K
    lora = tl.load(
        lora_ptr + offs, mask=k_mask, other=0.0, eviction_policy="evict_last"
    ).to(tl.float32)
    stream = tl.arange(0, HC_COUNT)
    row = stream * HC_DIMENSION + hidden
    weight = tl.load(
        weight_ptr + row[:, None] * K + offs[None, :],
        mask=k_mask[None, :],
        other=0.0,
        eviction_policy="evict_first",
    )
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    gate = gate.to(tl.float16).to(tl.float32)
    branch = tl.load(x_ptr + row).to(tl.float32)
    term = tl.sigmoid(gate) * branch
    result = tl.sum(tl.where(stream == 0, term, 0.0))
    result += tl.sum(tl.where(stream == 1, term, 0.0))
    result += tl.sum(tl.where(stream == 2, term, 0.0))
    result += tl.sum(tl.where(stream == 3, term, 0.0))
    tl.store(out_ptr + hidden, result / HC_COUNT)


def _runtime_ok(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> bool:
    return bool(
        x.ndim == 2
        and x.shape == (1, _HC_HIDDEN)
        and tuple(down_weight.shape)
        in ((_HC_RANK, _HC_HIDDEN), (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN))
        and up_weight.shape == (_HC_HIDDEN, _HC_RANK)
        and x.dtype == torch.float16
        and down_weight.dtype == torch.float16
        and up_weight.dtype == torch.float16
        and x.is_cuda
        and down_weight.is_cuda
        and up_weight.is_cuda
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and up_weight.is_contiguous()
        and x.device == down_weight.device == up_weight.device
    )


def _qwen38_sm70_fp16_fused_hc(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _runtime_ok(x, down_weight, up_weight):
        # Preserve the ordinary projection and FP16 materialization boundaries
        # for prefill and any unsupported runtime shape. This fallback lives
        # inside the opaque op so a prefill-first dynamic compile cannot bake
        # the M > 1 decision into subsequent decode graphs.
        down_and_injection = torch.nn.functional.linear(x, down_weight)
        lora = torch.ops.vllm.qwen4_exp_hc_silu(
            down_and_injection[..., :_HC_RANK], _HC_COUNT
        )
        injection = down_and_injection[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        gate = torch.nn.functional.linear(lora, up_weight)
        block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, _HC_COUNT)
        return block, injection
    try:
        from vllm.distributed.parallel_state import get_tp_group

        device_communicator = get_tp_group().device_communicator
        custom_ar = getattr(device_communicator, "ca_comm", None)
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        custom_ar = None

    if custom_ar is not None and custom_ar.can_sm70_qwen38_hc_shard(x):
        tp_rank = int(custom_ar.rank)
        local_down = x.new_empty((1, 88))
        gathered_down = x.new_empty((1, 336))
        block = x.new_empty((1, _HC_DIM))
        _qwen38_hc_down_local_shard_kernel[(88,)](
            x,
            down_weight,
            local_down,
            TP_RANK=tp_rank,
            num_warps=4,
        )
        custom_ar.sm70_qwen38_hc_down_allgather(local_down, gathered_down)
        if custom_ar.supports_sm70_qwen38_hc_up_mix_allgather():
            custom_ar.sm70_qwen38_hc_up_mix_allgather(
                gathered_down, up_weight, x, block
            )
            logger.info_once(
                "SM70 Qwen3.8 exact TP4 fused FP16 HC up/mix/gather enabled."
            )
            return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        if custom_ar.supports_sm70_qwen38_hc_output_allgather():
            local_block = x.new_empty((1, _HC_DIM // _HC_COUNT))
            _qwen38_hc_up_hidden_shard_kernel[(320,)](
                gathered_down,
                up_weight,
                x,
                local_block,
                TP_RANK=tp_rank,
                num_warps=8,
            )
            custom_ar.sm70_qwen38_hc_output_allgather(local_block, block)
            logger.info_once(
                "SM70 Qwen3.8 exact TP4 hidden-sharded FP16 HC route enabled."
            )
            return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]

        # An older wheel/sidecar can still use the established gate-sharded
        # route. Never pass its opaque communicator to a different DSO.
        local_gate = x.new_empty((1, _HC_DIM))
        _qwen38_hc_up_local_gate_kernel[(triton.cdiv(_HC_DIM, 8),)](
            gathered_down,
            up_weight,
            local_gate,
            TP_RANK=tp_rank,
            BLOCK_N=8,
            num_warps=8,
        )
        custom_ar.sm70_qwen38_hc_gate_mix(local_gate, x, block)
        logger.info_once(
            "SM70 Qwen3.8 exact TP4-sharded checkpoint-FP16 HC route enabled."
        )
        return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]

    lora = x.new_empty((1, _HC_RANK))
    injection = x.new_empty((1, _HC_COUNT))
    block = x.new_empty((1, _HC_DIM))
    _qwen38_hc_down_silu_inject_kernel[(_HC_RANK + _HC_COUNT,)](
        x,
        down_weight,
        lora,
        injection,
        K=_HC_HIDDEN,
        BLOCK_K=256,
        RANK_VALUE=_HC_RANK,
        HC_COUNT=_HC_COUNT,
        num_warps=4,
    )
    if _HC_UP_STREAM4:
        # P3a valve (VLLM_SM70_HC_UP_STREAM4=1): bit-identical to the row4
        # kernel (torch.equal on the unit check), one reduce chain per program
        # instead of four.
        _qwen38_hc_up_gate_mix_stream4_kernel[(triton.cdiv(_HC_DIM, 1),)](
            lora,
            up_weight,
            x,
            block,
            K=_HC_RANK,
            HC_DIMENSION=_HC_DIM,
            HC_COUNT=_HC_COUNT,
            BLOCK_K=512,
            num_warps=_HC_UP_STREAM4_WARPS,
        )
        logger.info_once(
            "SM70 Qwen3.8 fused checkpoint-FP16 HC M=1 route enabled "
            "(up-gate stream4 tile, warps=%d).", _HC_UP_STREAM4_WARPS
        )
    else:
        _qwen38_hc_up_gate_mix_row4_kernel[(triton.cdiv(_HC_DIM, 4),)](
            lora,
            up_weight,
            x,
            block,
            K=_HC_RANK,
            HC_DIMENSION=_HC_DIM,
            HC_COUNT=_HC_COUNT,
            BLOCK_N=4,
            BLOCK_K=512,
            num_warps=8,
        )
        logger.info_once(
            "SM70 Qwen3.8 fused checkpoint-FP16 HC M=1 route enabled.")
    return block, injection


def _qwen38_sm70_fp16_fused_hc_fake(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del down_weight, up_weight
    return (
        x.new_empty((*x.shape[:-1], _HC_DIM)),
        x.new_empty((*x.shape[:-1], _HC_COUNT)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_fused_hc",
    op_func=_qwen38_sm70_fp16_fused_hc,
    fake_impl=_qwen38_sm70_fp16_fused_hc_fake,
)


def maybe_apply_qwen38_sm70_fp16_fused_hc(
    down_layer: nn.Module,
    up_layer: nn.Module,
    x: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not enabled:
        return None
    down_weight = getattr(down_layer, "weight", None)
    up_weight = getattr(up_layer, "weight", None)
    if down_weight is None or up_weight is None:
        return None
    if down_weight.shape != (
        _HC_RANK + _HC_COUNT + 12,
        _HC_HIDDEN,
    ) or up_weight.shape != (_HC_HIDDEN, _HC_RANK):
        return None
    return torch.ops.vllm.qwen38_sm70_fp16_fused_hc(x, down_weight, up_weight)


def enable_qwen38_sm70_fp16_fused_hc(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Mark exact base-model HC modules for the fused M=1 route."""
    if (
        not envs.VLLM_SM70_QWEN38_FUSED_HC_FP16
        or dtype != torch.float16
        or not current_platform.is_device_capability((7, 0))
        or not _exact_runtime_contract(vllm_config)
    ):
        return

    enabled_count = 0
    for child in module.modules():
        if not (
            getattr(child, "use_combine", False)
            and _hc_lowrank_of(child) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
            and hasattr(child, "input_mix_weight_down_block_inject")
            and hasattr(child, "input_mix_weight_up")
        ):
            continue
        child._sm70_qwen38_fp16_fused_hc = True
        enabled_count += 1

    if enabled_count:
        logger.info_once(
            "Prepared %d Qwen3.8 SM70 fused checkpoint-FP16 HC modules.",
            enabled_count,
        )


__all__ = [
    "enable_qwen38_sm70_fp16_fused_hc",
    "maybe_apply_qwen38_sm70_fp16_fused_hc",
]


# ---------------------------------------------------------------------------
# up-projection + gate-mix fusion (TP-agnostic, no collective).
#
# The baseline chain per HC module is four launches:
#   down GEMV -> hc_silu -> up GEMV -> hc_gate_mix
# where the last two move a 10240-wide intermediate through HBM. Each of them is
# a ~13 us latency floor at M=1, so the chain costs ~58 us per module while the
# bytes involved only justify ~30 us. Adding the mix into the up GEMV removes
# one launch and the intermediate: one program per output column computes the
# four dots it needs and finishes the sigmoid mix.
# ---------------------------------------------------------------------------
@triton.jit
def _qwen38_hc_up_gate_kernel(
    lora_ptr,
    up_ptr,
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    column = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    rmask = r < RANK
    lora0 = tl.load(lora_ptr + r, mask=rmask, other=0.0).to(tl.float32)
    if ROWS == 1:
        lora1 = lora0
        lora2 = lora0
    else:
        lora1 = tl.load(lora_ptr + RANK + r, mask=rmask, other=0.0).to(tl.float32)
        if ROWS > 2:
            lora2 = tl.load(lora_ptr + 2 * RANK + r, mask=rmask, other=0.0).to(
                tl.float32
            )
        else:
            lora2 = lora0
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    for stream in tl.static_range(COUNT):
        weight = tl.load(
            up_ptr + (stream * DIM + column) * RANK + r,
            mask=rmask,
            other=0.0,
        ).to(tl.float32)
        # One weight read feeds every verified row; keep the baseline FP16 gate
        # boundary before the sigmoid.
        gate0 = tl.sum(lora0 * weight, axis=0).to(tl.float16).to(tl.float32)
        acc0 += tl.sigmoid(gate0) * tl.load(
            x_ptr + stream * DIM + column
        ).to(tl.float32)
        if ROWS > 1:
            gate1 = tl.sum(lora1 * weight, axis=0).to(tl.float16).to(tl.float32)
            acc1 += tl.sigmoid(gate1) * tl.load(
                x_ptr + (COUNT + stream) * DIM + column
            ).to(tl.float32)
        if ROWS > 2:
            gate2 = tl.sum(lora2 * weight, axis=0).to(tl.float16).to(tl.float32)
            acc2 += tl.sigmoid(gate2) * tl.load(
                x_ptr + (2 * COUNT + stream) * DIM + column
            ).to(tl.float32)
    tl.store(out_ptr + column, (acc0 / COUNT).to(tl.float16))
    if ROWS > 1:
        tl.store(out_ptr + DIM + column, (acc1 / COUNT).to(tl.float16))
    if ROWS > 2:
        tl.store(out_ptr + 2 * DIM + column, (acc2 / COUNT).to(tl.float16))


@triton.jit
def _qwen38_hc_up_gate_wide_kernel(
    lora_ptr,
    up_ptr,
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    COUNT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Same maths as _qwen38_hc_up_gate_kernel, BLOCK_C columns per program.

    One column per program left each lane holding two or three weight scalars,
    so the 6.5 MB of weights per call streamed at 42% of peak. Consecutive
    columns are contiguous blocks of RANK, so a BLOCK_C-wide tile is one
    coalesced run and `lora` is loaded once for all of them.
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < DIM
    r = tl.arange(0, BLOCK_R)
    rmask = r < RANK
    lora0 = tl.load(lora_ptr + r, mask=rmask, other=0.0).to(tl.float32)
    if ROWS == 1:
        lora1 = lora0
        lora2 = lora0
    else:
        lora1 = tl.load(lora_ptr + RANK + r, mask=rmask, other=0.0).to(tl.float32)
        if ROWS > 2:
            lora2 = tl.load(lora_ptr + 2 * RANK + r, mask=rmask, other=0.0).to(
                tl.float32
            )
        else:
            lora2 = lora0
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for stream in tl.static_range(COUNT):
        w = tl.load(
            up_ptr + (stream * DIM + cols[:, None]) * RANK + r[None, :],
            mask=cmask[:, None] & rmask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate0 = tl.sum(lora0[None, :] * w, axis=1).to(tl.float16).to(tl.float32)
        acc0 += tl.sigmoid(gate0) * tl.load(
            x_ptr + stream * DIM + cols, mask=cmask, other=0.0
        ).to(tl.float32)
        if ROWS > 1:
            gate1 = tl.sum(lora1[None, :] * w, axis=1).to(tl.float16).to(tl.float32)
            acc1 += tl.sigmoid(gate1) * tl.load(
                x_ptr + (COUNT + stream) * DIM + cols, mask=cmask, other=0.0
            ).to(tl.float32)
        if ROWS > 2:
            gate2 = tl.sum(lora2[None, :] * w, axis=1).to(tl.float16).to(tl.float32)
            acc2 += tl.sigmoid(gate2) * tl.load(
                x_ptr + (2 * COUNT + stream) * DIM + cols, mask=cmask, other=0.0
            ).to(tl.float32)
    tl.store(out_ptr + cols, (acc0 / COUNT).to(tl.float16), mask=cmask)
    if ROWS > 1:
        tl.store(out_ptr + DIM + cols, (acc1 / COUNT).to(tl.float16), mask=cmask)
    if ROWS > 2:
        tl.store(out_ptr + 2 * DIM + cols, (acc2 / COUNT).to(tl.float16), mask=cmask)


def _qwen38_sm70_fp16_hc_up_gate(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    lora: torch.Tensor,
) -> torch.Tensor:
    rows = int(x.shape[0])
    out = x.new_empty((rows, _HC_DIM))
    if _HC_UP_WIDE > 1:
        _qwen38_hc_up_gate_wide_kernel[(triton.cdiv(_HC_DIM, _HC_UP_WIDE),)](
            lora,
            up_weight,
            x,
            out,
            ROWS=rows,
            RANK=_HC_RANK,
            DIM=_HC_DIM,
            COUNT=_HC_COUNT,
            BLOCK_R=512,
            BLOCK_C=_HC_UP_WIDE,
            num_warps=4,
        )
        return out
    _qwen38_hc_up_gate_kernel[(_HC_DIM,)](
        lora,
        up_weight,
        x,
        out,
        ROWS=rows,
        RANK=_HC_RANK,
        DIM=_HC_DIM,
        COUNT=_HC_COUNT,
        BLOCK_R=512,
        num_warps=4,
    )
    return out


def _qwen38_sm70_fp16_hc_up_gate_fake(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    lora: torch.Tensor,
) -> torch.Tensor:
    del up_weight, lora
    return x.new_empty((x.shape[0], _HC_DIM))


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_hc_up_gate",
    op_func=_qwen38_sm70_fp16_hc_up_gate,
    fake_impl=_qwen38_sm70_fp16_hc_up_gate_fake,
)


def _up_gate_runtime_ok(
    x: torch.Tensor, up_weight: torch.Tensor, lora: torch.Tensor
) -> bool:
    rows = x.shape[0] if x.dim() == 2 else -1
    return bool(
        rows in (1, 2, 3)
        and x.shape == (rows, _HC_HIDDEN)
        and lora.shape == (rows, _HC_RANK)
        and up_weight.shape == (_HC_HIDDEN, _HC_RANK)
        and x.dtype == torch.float16
        and lora.dtype == torch.float16
        and up_weight.dtype == torch.float16
        and x.is_contiguous()
        and lora.is_contiguous()
        and up_weight.is_contiguous()
        and x.is_cuda
        and lora.device == x.device
        and up_weight.device == x.device
    )


def maybe_apply_qwen38_sm70_fp16_hc_up_gate(
    up_weight: torch.Tensor,
    x: torch.Tensor,
    lora: torch.Tensor,
    enabled: bool,
) -> torch.Tensor | None:
    """Fused up-projection + gate-mix for one M=1 HC module, or None."""
    if not enabled or not _up_gate_runtime_ok(x, up_weight, lora):
        return None
    return torch.ops.vllm.qwen38_sm70_fp16_hc_up_gate(x, up_weight, lora)


def _rebind_hc_dims(module: nn.Module) -> None:
    """Derive the per-rank HC dims from the runtime layout (TP-dependent).

    _HC_DIM/_HC_RANK/_HC_HIDDEN above are the TP8 layout. Under TP4 every HC
    module is twice as wide, so the shape checks below reject every module and
    the fused path silently stays off. Rebinding from the first real HC module
    keeps one source of truth for both TP sizes (the kernels take these as
    constexpr arguments, so Triton compiles one variant per distinct value).
    """
    global _HC_DIM, _HC_RANK, _HC_HIDDEN
    for child in module.modules():
        up = getattr(getattr(child, "input_mix_weight_up", None), "weight", None)
        dim = getattr(child, "hidden_size", None)
        rank = _hc_lowrank_of(child)
        if not (isinstance(up, torch.Tensor) and dim and rank
                and getattr(child, "hc_count", None) == _HC_COUNT
                and tuple(up.shape) == (_HC_COUNT * int(dim), int(rank))):
            continue
        if (int(dim), int(rank)) != (_HC_DIM, _HC_RANK):
            _HC_DIM, _HC_RANK = int(dim), int(rank)
            _HC_HIDDEN = _HC_COUNT * _HC_DIM
            logger.info_once(
                "SM70 HC dims rebound to the runtime layout: dim=%d rank=%d "
                "hidden=%d (TP-dependent)", _HC_DIM, _HC_RANK, _HC_HIDDEN)
        return


def enable_qwen38_sm70_fp16_hc_up_gate(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Mark HC modules whose up-projection can use the fused up+gate kernel."""
    if not envs.VLLM_SM70_QWEN38_FUSED_HC_UP_GATE:
        return
    if dtype != torch.float16 or not current_platform.is_device_capability((7, 0)):
        return
    _rebind_hc_dims(module)
    enabled = 0
    for child in module.modules():
        if not (
            _hc_lowrank_of(child) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
        ):
            continue
        up_weight = getattr(getattr(child, "input_mix_weight_up", None), "weight", None)
        if not (isinstance(up_weight, torch.Tensor)
                and tuple(up_weight.shape) == (_HC_HIDDEN, _HC_RANK)
                and up_weight.dtype == torch.float16):
            continue
        child._sm70_qwen38_fp16_up_gate = True
        enabled += 1
    if enabled:
        logger.info_once(
            "Prepared %d Qwen3.8 SM70 fused FP16 HC up+gate modules.", enabled
        )
    else:
        logger.warning_once(
            "Qwen3.8 fused FP16 HC up+gate opt-in found no targets."
        )


# ---------------------------------------------------------------------------
# down-projection + SiLU fusion.
#
# The baseline issues a cuBLAS down GEMV (336 x 10240) and then a separate
# hc_silu launch over the 320 lora rows. Folding the SiLU into the GEMV store
# removes one launch and one round trip of the 320-row intermediate; the writes
# are already FP16-materialised exactly as the baseline op chain does.
# ---------------------------------------------------------------------------
@triton.jit
def _qwen38_hc_down_silu_kernel(
    x_ptr,
    w_ptr,
    lora_ptr,
    injection_ptr,
    ROWS: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    RANK: tl.constexpr,
    COUNT: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    is_lora = row < RANK
    if HAS_INJECT:
        is_injection = (row >= RANK) & (row < RANK + COUNT)
    else:
        is_injection = row < 0
    active = is_lora | is_injection
    off = tl.arange(0, BLOCK)
    acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in tl.static_range(0, K, BLOCK):
        idx = start + off
        mask = active & (idx < K)
        w = tl.load(w_ptr + row * K + idx, mask=mask, other=0.0).to(tl.float32)
        acc0 += tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32) * w
        if ROWS > 1:
            acc1 += (
                tl.load(x_ptr + K + idx, mask=mask, other=0.0).to(tl.float32) * w
            )
        if ROWS > 2:
            acc2 += (
                tl.load(x_ptr + 2 * K + idx, mask=mask, other=0.0).to(tl.float32)
                * w
            )
    value0 = tl.sum(acc0, axis=0).to(tl.float16).to(tl.float32)
    scaled0 = value0 / COUNT
    tl.store(lora_ptr + row, (scaled0 * tl.sigmoid(scaled0)).to(tl.float16),
             mask=is_lora)
    tl.store(injection_ptr + (row - RANK), value0.to(tl.float16),
             mask=is_injection)
    if ROWS > 1:
        value1 = tl.sum(acc1, axis=0).to(tl.float16).to(tl.float32)
        scaled1 = value1 / COUNT
        tl.store(lora_ptr + RANK + row,
                 (scaled1 * tl.sigmoid(scaled1)).to(tl.float16), mask=is_lora)
        tl.store(injection_ptr + COUNT + (row - RANK), value1.to(tl.float16),
                 mask=is_injection)
    if ROWS > 2:
        value2 = tl.sum(acc2, axis=0).to(tl.float16).to(tl.float32)
        scaled2 = value2 / COUNT
        tl.store(lora_ptr + 2 * RANK + row,
                 (scaled2 * tl.sigmoid(scaled2)).to(tl.float16), mask=is_lora)
        tl.store(injection_ptr + 2 * COUNT + (row - RANK), value2.to(tl.float16),
                 mask=is_injection)


def _qwen38_sm70_fp16_hc_down_silu(
    x: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(x.shape[0])
    has_inject = down_weight.shape[0] > _HC_RANK
    lora = x.new_empty((rows, _HC_RANK))
    injection = x.new_empty((rows, _HC_COUNT if has_inject else 1))
    _qwen38_hc_down_silu_kernel[(_HC_RANK + (_HC_COUNT if has_inject else 0),)](
        x,
        down_weight,
        lora,
        injection,
        ROWS=rows,
        HAS_INJECT=has_inject,
        RANK=_HC_RANK,
        COUNT=_HC_COUNT,
        K=_HC_HIDDEN,
        BLOCK=2048,
        num_warps=8,
    )
    return lora, injection


def _qwen38_sm70_fp16_hc_down_silu_fake(
    x: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del down_weight
    return x.new_empty((x.shape[0], _HC_RANK)), x.new_empty((x.shape[0], _HC_COUNT))


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_hc_down_silu",
    op_func=_qwen38_sm70_fp16_hc_down_silu,
    fake_impl=_qwen38_sm70_fp16_hc_down_silu_fake,
)


def maybe_apply_qwen38_sm70_fp16_hc_down_silu(
    down_weight: torch.Tensor,
    x: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused down-projection + SiLU for one M=1 HC module, or None."""
    if not enabled:
        return None
    rows = x.shape[0] if x.dim() == 2 else -1
    if not (
        rows in (1, 2, 3)
        and x.shape == (rows, _HC_HIDDEN)
        and down_weight.shape == (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN)
        and x.dtype == torch.float16
        and down_weight.dtype == torch.float16
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and x.is_cuda
        and down_weight.device == x.device
    ):
        return None
    return torch.ops.vllm.qwen38_sm70_fp16_hc_down_silu(x, down_weight)


def enable_qwen38_sm70_fp16_hc_down_silu(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Mark HC modules whose down-projection can use the fused down+SiLU kernel."""
    del vllm_config
    if not envs.VLLM_SM70_QWEN38_FUSED_HC_DOWN_SILU:
        return
    if dtype != torch.float16 or not current_platform.is_device_capability((7, 0)):
        return
    _rebind_hc_dims(module)
    enabled = 0
    for child in module.modules():
        if not (
            _hc_lowrank_of(child) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
        ):
            continue
        down_weight = getattr(
            getattr(child, "input_mix_weight_down_block_inject", None), "weight", None
        )
        if down_weight is None:
            down_weight = getattr(
                getattr(child, "input_mix_weight_down", None), "weight", None
            )
        if not (
            isinstance(down_weight, torch.Tensor)
            and tuple(down_weight.shape)
            in ((_HC_RANK, _HC_HIDDEN), (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN))
            and down_weight.dtype == torch.float16
        ):
            continue
        child._sm70_qwen38_fp16_down_silu = True
        enabled += 1
    if enabled:
        logger.info_once(
            "Prepared %d Qwen3.8 SM70 fused FP16 HC down+SiLU modules.", enabled
        )
    else:
        logger.warning_once(
            "Qwen3.8 fused FP16 HC down+SiLU opt-in found no targets."
        )
