# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in checkpoint-FP16 Qwen3.8 decode GEMV kernels for SM70.

The route is deliberately narrow: exact Qwen3.8 Flash Next topology, TP4 or
TP8, no speculative decoding, FP16 checkpoint weights, and one decode token.
All prefill and unsupported shapes retain the ordinary unquantized linear path.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from torch import nn

import vllm.envs as envs
def use_sm70_decode_graph_semantics() -> bool:
    # 1Cat gates the M=1 branch on their dual-compile lane; this tree has
    # no such lane, so the branch is always live.
    return True
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class _GemvPlan(NamedTuple):
    block_k: int
    num_warps: int
    load_policy: int


_ROUTE_LOG = os.environ.get("VLLM_SM70_ROUTE_LOG", "0") == "1"
_ROUTE_NOTES: dict[str, int] = {}


def _route_note(role: str, shape, rows: int, hit: bool, why: str) -> None:
    """Diagnostic (default off): which projections our GEMV route serves."""
    if not _ROUTE_LOG:
        return
    key = f"{str(role)[-36:]} {shape} rows={rows} {why} hit={hit}"
    _ROUTE_NOTES[key] = _ROUTE_NOTES.get(key, 0) + 1
    if sum(_ROUTE_NOTES.values()) % 500 == 0:
        import json

        try:
            with open(f"/tmp/sm70_route_{os.getpid()}.json", "w") as fh:
                json.dump(_ROUTE_NOTES, fh, indent=1)
        except OSError:
            pass


# Single-token decode plus the two-row speculative verify.
_MAX_DECODE_ROWS = 3

_HC_DOWN_SUFFIX = ".input_mix_weight_down_block_inject"
_GDN_QKVZ_SUFFIX = ".linear_attn.in_proj_qkvz"
_GDN_BA_SUFFIX = ".linear_attn.in_proj_ba"
_GDN_OUT_SUFFIX = ".linear_attn.out_proj"
_QSA_QKV_SUFFIX = ".self_attn.qkv_proj"
_QSA_OUT_SUFFIX = ".self_attn.o_proj"
_QSA_INDEX_SUFFIX = ".self_attn.indexer.index_qk_proj"
_ROUTER_SUFFIX = ".mlp.gate"

# Plans are cold-cache CUDA Graph winners on real checkpoint weights. Keep the
# role in the key: GDN and QSA can share a physical shape while remaining
# independently auditable.
_ROLE_PLANS: tuple[tuple[str, tuple[int, int], _GemvPlan], ...] = (
    (_HC_DOWN_SUFFIX, (336, 10240), _GemvPlan(512, 4, 1)),
    (_GDN_QKVZ_SUFFIX, (4096, 2560), _GemvPlan(512, 2, 0)),
    (_GDN_BA_SUFFIX, (24, 2560), _GemvPlan(512, 4, 0)),
    (_GDN_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 1)),
    (_QSA_QKV_SUFFIX, (3584, 2560), _GemvPlan(512, 2, 0)),
    (_QSA_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 0)),
    (_QSA_INDEX_SUFFIX, (640, 2560), _GemvPlan(512, 2, 0)),
    (_ROUTER_SUFFIX, (512, 2560), _GemvPlan(1024, 8, 0)),
    # TP8 per-rank layouts actually instantiated by the 8-way serve. GDN rows
    # halve (4096 -> 2048); the HC/up-shaped 320-row reduce loses to cuBLAS at
    # this K, so it stays unplanned on purpose.
    (_GDN_QKVZ_SUFFIX, (2048, 2560), _GemvPlan(1024, 8, 0)),
    # TP8 in_proj_ba is only 12 rows; cuBLAS picks a single-CTA kernel
    # there (measured 38 launches/step at ~56 us each).
    (_GDN_BA_SUFFIX, (12, 2560), _GemvPlan(512, 2, 0)),
    (".mlp.shared_expert.gate_up_proj", (160, 2560), _GemvPlan(512, 4, 0)),
    (".mlp.shared_expert.down_proj", (2560, 80), _GemvPlan(1024, 8, 0)),
    # TP8 shared-expert gate is a (1, 2560) dot: cuBLAS spends two kernels
    # (dot + reduce, ~6.8 us/step total) where one row-GEMV suffices.
    (".mlp.shared_expert_gate", (1, 2560), _GemvPlan(2048, 4, 0)),
    # MTP draft (1 layer): these four shapes had no plan, so the draft's
    # projections fell back to cuBLAS (~22 us/GEMV instead of ~6 us).
    (".self_attn.qkv_proj", (2688, 2560), _GemvPlan(512, 2, 0)),
    (".self_attn.o_proj", (2560, 768), _GemvPlan(512, 4, 0)),
    ("fc_embedding", (320, 2560), _GemvPlan(512, 4, 0)),
    ("fc_hidden", (320, 2560), _GemvPlan(512, 4, 0)),
    # TP4 per-rank layouts (2026-10-24): the table above is TP8-only, so a TP4
    # serve prepared 312 of 384 projections and silently left the rest on
    # cuBLAS. Sharded dims double relative to the TP8 entries; plans mirror the
    # TP8 choice for the same role (block_k/warps are validated in a TP4 window).
    (_GDN_QKVZ_SUFFIX, (4096, 2560), _GemvPlan(512, 2, 0)),
    (_GDN_BA_SUFFIX, (24, 2560), _GemvPlan(512, 4, 0)),
    (_GDN_OUT_SUFFIX, (2560, 3072), _GemvPlan(512, 4, 1)),
    (_QSA_QKV_SUFFIX, (7168, 2560), _GemvPlan(512, 2, 0)),
    (_QSA_OUT_SUFFIX, (2560, 3072), _GemvPlan(512, 4, 0)),
    (_QSA_INDEX_SUFFIX, (1280, 2560), _GemvPlan(512, 2, 0)),
    (_ROUTER_SUFFIX, (1024, 2560), _GemvPlan(1024, 8, 0)),
    (".mlp.shared_expert.gate_up_proj", (320, 2560), _GemvPlan(512, 4, 0)),
    (".mlp.shared_expert.down_proj", (2560, 160), _GemvPlan(1024, 8, 0)),
    (".self_attn.qkv_proj", (5376, 2560), _GemvPlan(512, 2, 0)),
    (".self_attn.o_proj", (2560, 1536), _GemvPlan(512, 4, 0)),
    ("fc_embedding", (640, 2560), _GemvPlan(512, 4, 0)),
    ("fc_hidden", (640, 2560), _GemvPlan(512, 4, 0)),
)

# Retain the legacy two-argument custom-op behavior for external callers.
# Model layers pass their role explicitly: same-shape GDN/QSA plans differ.
_SHAPE_PLANS = {shape: plan for _, shape, plan in _ROLE_PLANS}


@triton.jit
def _qwen38_gdn_projection_split_kernel(
    qkvz,
    ba,
    qkv,
    z,
    b,
    a,
    QKV: tl.constexpr,
    Z: tl.constexpr,
    B: tl.constexpr,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
    tl.store(qkv + row * QKV + col, value, col < QKV)
    tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
    if tl.program_id(1) == 0:
        tl.static_assert(B + A <= BLOCK)
        gate_col = tl.arange(0, BLOCK)
        gate = tl.load(ba + row * (B + A) + gate_col, gate_col < B + A, other=0)
        tl.store(b + row * B + gate_col, gate, gate_col < B)
        tl.store(a + row * A + gate_col - B, gate, (gate_col >= B) & (gate_col < B + A))


def _split_gdn_projection_outputs(qkvz, ba):
    m = qkvz.shape[0]
    out = tuple(qkvz.new_empty((m, n)) for n in (2560, 1536, 12, 12))
    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
        qkvz,
        ba,
        *out,
        QKV=2560,
        Z=1536,
        B=12,
        A=12,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return out


def _can_fuse_gdn_projection_split(qkvz: torch.Tensor, ba: torch.Tensor) -> bool:
    # This copy-only operation does not change either GEMM. Keep the existing
    # M1 path and all unsupported layouts; no maximum batch/sequence binding.
    return bool(
        envs.VLLM_SM70_GDN_BATCH_SPLIT_COPY
        and _is_packed_row_major(qkvz)
        and _is_packed_row_major(ba)
        and qkvz.shape[0] > 1
        and qkvz.shape[1] == 4096
        and ba.shape == (qkvz.shape[0], 24)
        and qkvz.dtype == ba.dtype == torch.float16
        and qkvz.is_cuda
        and ba.is_cuda
        and qkvz.device == ba.device
        and current_platform.is_device_capability(70)
    )


@triton.jit
def _qwen38_fp16_row_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
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
        else:
            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        acc += x.to(tl.float32) * weight.to(tl.float32)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _qwen38_fp16_row_gemv2_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
):
    """Two-token verify variant: one weight read feeds both rows.

    Speculative verify runs M=2 (the drafted token plus the bonus row) and the
    projection is weight-bound, so the second row is nearly free once it shares
    the load instead of falling back to a generic M=2 GEMM.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        weight = weight.to(tl.float32)
        x0 = tl.load(x_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + K + indices, mask=mask, other=0.0).to(tl.float32)
        acc0 += x0 * weight
        acc1 += x1 * weight
    tl.store(out_ptr + row, tl.sum(acc0, axis=0))
    tl.store(out_ptr + tl.num_programs(0) + row, tl.sum(acc1, axis=0))


@triton.jit
def _qwen38_fp16_row_gemv3_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
):
    """Three-row speculative verify: one weight read feeds all three rows.

    k=2 is a measured net loss today (13.1 vs 12.1 ms per token: the third row
    costs ~7 ms while adding 0.3 tokens per step), so this kernel only runs when
    k=2 is re-enabled for the row-batching work; it is kept because the M=3
    arithmetic is verified bitwise against per-row M=1.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        weight = weight.to(tl.float32)
        acc0 += tl.load(x_ptr + indices, mask=mask, other=0.0).to(tl.float32) * weight
        acc1 += tl.load(x_ptr + K + indices, mask=mask, other=0.0).to(
            tl.float32
        ) * weight
        acc2 += tl.load(x_ptr + 2 * K + indices, mask=mask, other=0.0).to(
            tl.float32
        ) * weight
    tl.store(out_ptr + row, tl.sum(acc0, axis=0))
    tl.store(out_ptr + N + row, tl.sum(acc1, axis=0))
    tl.store(out_ptr + 2 * N + row, tl.sum(acc2, axis=0))


@triton.jit
def _qwen38_fp16_gdn_input_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    is_qkvz = row < 4096
    ba_row = row - 4096
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        acc += x.to(tl.float32) * weight.to(tl.float32)

    value = tl.sum(acc, axis=0)
    is_qkv = is_qkvz & (row < 2560)
    is_z = is_qkvz & (row >= 2560)
    is_b = (~is_qkvz) & (ba_row < 12)
    is_a = (~is_qkvz) & (ba_row >= 12)
    tl.store(qkv_out_ptr + row, value, mask=is_qkv)
    tl.store(z_out_ptr + row - 2560, value, mask=is_z)
    tl.store(b_out_ptr + ba_row, value, mask=is_b)
    tl.store(a_out_ptr + ba_row - 12, value, mask=is_a)


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return bool(
        not envs.VLLM_BATCH_INVARIANT
        and x.shape[0] <= _MAX_DECODE_ROWS
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _qwen38_sm70_fp16_gemv(
    x: torch.Tensor, weight: torch.Tensor, role: str = ""
) -> torch.Tensor:
    shape = (weight.shape[0], weight.shape[1])
    plan = _plan_for(role, shape) if role else _SHAPE_PLANS.get(shape)
    if plan is None or not _runtime_ok(x, weight):
        if _ROUTE_LOG:
            _route_note(role, shape, int(x.shape[0]) if x.dim() else -1,
                        False, "no-plan" if plan is None else "runtime-ok-false")
        return torch.nn.functional.linear(x, weight)
    if _ROUTE_LOG:
        _route_note(role, shape, int(x.shape[0]), True, "hit")

    rows = int(x.shape[0])
    out = torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=x.device)
    if rows == 1:
        _qwen38_fp16_row_gemv_kernel[(weight.shape[0],)](
            x, weight, out,
            K=weight.shape[1],
            BLOCK_K=plan.block_k,
            LOAD_POLICY=plan.load_policy,
            num_warps=plan.num_warps,
        )
    elif rows == 2:
        _qwen38_fp16_row_gemv2_kernel[(weight.shape[0],)](
            x, weight, out,
            K=weight.shape[1],
            BLOCK_K=plan.block_k,
            LOAD_POLICY=plan.load_policy,
            num_warps=plan.num_warps,
        )
    else:
        _qwen38_fp16_row_gemv3_kernel[(weight.shape[0],)](
            x, weight, out,
            N=weight.shape[0],
            K=weight.shape[1],
            BLOCK_K=plan.block_k,
            LOAD_POLICY=plan.load_policy,
            num_warps=plan.num_warps,
        )
    logger.info_once("SM70 Qwen3.8 checkpoint-FP16 M=1 GEMV route enabled.")
    return out


def _qwen38_sm70_fp16_gemv_fake(
    x: torch.Tensor, weight: torch.Tensor, role: str = ""
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gemv",
    op_func=_qwen38_sm70_fp16_gemv,
    fake_impl=_qwen38_sm70_fp16_gemv_fake,
)


def _qwen38_sm70_fp16_gdn_input(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (
        qkvz_weight.shape == (4096, 2560)
        and ba_weight.shape == (24, 2560)
        and _runtime_ok(x, qkvz_weight)
        and _runtime_ok(x, ba_weight)
    ):
        qkvz = torch.nn.functional.linear(x, qkvz_weight)
        ba = torch.nn.functional.linear(x, ba_weight)
        if _can_fuse_gdn_projection_split(qkvz, ba):
            logger.info_once("SM70 GDN batched projection split-copy fusion enabled.")
            return _split_gdn_projection_outputs(qkvz, ba)
        return (
            qkvz[..., :2560].contiguous(),
            qkvz[..., 2560:].contiguous(),
            ba[..., :12].contiguous(),
            ba[..., 12:].contiguous(),
        )

    qkv = x.new_empty((1, 2560))
    z = x.new_empty((1, 1536))
    b = x.new_empty((1, 12))
    a = x.new_empty((1, 12))
    _qwen38_fp16_gdn_input_kernel[(4096 + 24,)](
        x,
        qkvz_weight,
        ba_weight,
        qkv,
        z,
        b,
        a,
        K=2560,
        BLOCK_K=512,
        num_warps=2,
    )
    logger.info_once("SM70 Qwen3.8 checkpoint-FP16 fused GDN input route enabled.")
    return qkv, z, b, a


@triton.jit
def _qwen38_fp16_gdn_input_packed_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkvz_out_ptr,
    ba_out_ptr,
    N_QKVZ: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Same merged qkvz+ba GEMV as _qwen38_fp16_gdn_input_kernel but stores
    # the packed (N_QKVZ, N_BA) layout our GDN forward consumes directly.
    row = tl.program_id(0)
    is_qkvz = row < N_QKVZ
    ba_row = row - N_QKVZ
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        acc += x.to(tl.float32) * weight.to(tl.float32)

    value = tl.sum(acc, axis=0)
    if is_qkvz:
        tl.store(qkvz_out_ptr + row, value)
    else:
        tl.store(ba_out_ptr + ba_row, value)


def _qwen38_sm70_fp16_gdn_input_packed(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_qkvz, n_ba = qkvz_weight.shape[0], ba_weight.shape[0]
    if not (
        qkvz_weight.shape[1] == 2560
        and ba_weight.shape[1] == 2560
        and (n_qkvz, n_ba) in ((4096, 24), (2048, 12))
        and _runtime_ok(x, qkvz_weight)
        and _runtime_ok(x, ba_weight)
    ):
        return (
            torch.nn.functional.linear(x, qkvz_weight),
            torch.nn.functional.linear(x, ba_weight),
        )

    qkvz = x.new_empty((1, n_qkvz))
    ba = x.new_empty((1, n_ba))
    # 512/2 wins on both layouts (TP4 4096+24 and TP8 2048+12): the extra warps
    # buy no latency here and only cost occupancy.
    _qwen38_fp16_gdn_input_packed_kernel[(n_qkvz + n_ba,)](
        x,
        qkvz_weight,
        ba_weight,
        qkvz,
        ba,
        N_QKVZ=n_qkvz,
        K=2560,
        BLOCK_K=512,
        num_warps=2,
    )
    logger.info_once(
        "SM70 Qwen3.8 checkpoint-FP16 fused GDN input (packed) route enabled."
    )
    return qkvz, ba


def _qwen38_sm70_fp16_gdn_input_packed_fake(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty((*x.shape[:-1], qkvz_weight.shape[0])),
        x.new_empty((*x.shape[:-1], ba_weight.shape[0])),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gdn_input_packed",
    op_func=_qwen38_sm70_fp16_gdn_input_packed,
    fake_impl=_qwen38_sm70_fp16_gdn_input_packed_fake,
)


def _qwen38_sm70_fp16_gdn_input_fake(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qkvz_weight, ba_weight
    batch_shape = x.shape[:-1]
    return (
        x.new_empty((*batch_shape, 2560)),
        x.new_empty((*batch_shape, 1536)),
        x.new_empty((*batch_shape, 12)),
        x.new_empty((*batch_shape, 12)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gdn_input",
    op_func=_qwen38_sm70_fp16_gdn_input,
    fake_impl=_qwen38_sm70_fp16_gdn_input_fake,
)


class Qwen38SM70FP16LinearMethod(UnquantizedLinearMethod):
    """Use the row-GEMV custom op for admitted single-token projections."""

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The first dynamic compile sees prefill (M > 1). Keep the M=1
        # decision inside the opaque op so decode does not inherit a baked-in
        # prefill branch.
        if bias is None and use_sm70_decode_graph_semantics():
            return torch.ops.vllm.qwen38_sm70_fp16_gemv(
                x, layer.weight, getattr(layer, "prefix", "")
            )
        return super().apply(layer, x, bias)


def _plan_for(prefix: str, shape: tuple[int, int]) -> _GemvPlan | None:
    for suffix, expected_shape, plan in _ROLE_PLANS:
        if prefix.endswith(suffix) and shape == expected_shape:
            return plan
    return None


def _exact_runtime_contract(vllm_config=None) -> bool:
    try:
        config = vllm_config or get_current_vllm_config()
        text_config = config.model_config.hf_text_config
        tp_size = int(config.parallel_config.tensor_parallel_size)
    except (AssertionError, AttributeError, RuntimeError):
        return False

    spec = config.speculative_config
    spec_ok = spec is None or int(getattr(spec, "num_speculative_tokens", 0) or 0) <= 2
    return bool(
        tp_size in (4, 8)
        and spec_ok
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


def enable_qwen38_sm70_fp16_gemv(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Replace admitted unquantized methods before checkpoint loading."""
    if envs.VLLM_SM70_GEMV_SHAPES:
        for child in module.modules():
            if not (
                isinstance(child, LinearBase)
                and type(child.quant_method) is UnquantizedLinearMethod
            ):
                continue
            weight = getattr(child, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            prefix = str(getattr(child, "prefix", ""))
            shape = (int(weight.shape[0]), int(weight.shape[1]))
            logger.info(
                "SM70 GEMV shape: %s %s plan=%s",
                prefix[-64:],
                shape,
                _plan_for(prefix, shape),
            )
    if not envs.VLLM_SM70_QWEN38_FP16_GEMV:
        return
    capability_ok = current_platform.is_device_capability((7, 0))
    contract_ok = _exact_runtime_contract(vllm_config)
    if dtype != torch.float16 or not capability_ok or not contract_ok:
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in rejected: "
            "dtype=%s sm70=%s exact_contract=%s.",
            dtype,
            capability_ok,
            contract_ok,
        )
        return

    replaced = 0
    for child in module.modules():
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        if _plan_for(str(getattr(child, "prefix", "")), shape) is None:
            continue
        child.quant_method = Qwen38SM70FP16LinearMethod()
        replaced += 1

    fused_gdn_inputs = 0
    if envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        for child in module.modules():
            qkvz = getattr(child, "in_proj_qkvz", None)
            ba = getattr(child, "in_proj_ba", None)
            qkvz_weight = getattr(qkvz, "weight", None)
            ba_weight = getattr(ba, "weight", None)
            tp_size = int(getattr(child, "tp_size", 0) or 0)
            key_dim = int(getattr(child, "key_dim", 0) or 0)
            value_dim = int(getattr(child, "value_dim", 0) or 0)
            num_v_heads = int(getattr(child, "num_v_heads", 0) or 0)
            if tp_size <= 0 or key_dim <= 0 or value_dim <= 0 or num_v_heads <= 0:
                continue
            expected_qkvz = (2 * key_dim + 2 * value_dim) // tp_size
            expected_ba = 2 * (num_v_heads // tp_size)
            if not (
                isinstance(qkvz_weight, torch.Tensor)
                and tuple(qkvz_weight.shape) == (expected_qkvz, 2560)
                and isinstance(ba_weight, torch.Tensor)
                and tuple(ba_weight.shape) == (expected_ba, 2560)
                and isinstance(
                    getattr(qkvz, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and isinstance(
                    getattr(ba, "quant_method", None),
                    (Qwen38SM70FP16LinearMethod, UnquantizedLinearMethod),
                )
                and not bool(getattr(child, "gqa_interleaved_layout", True))
                and not bool(getattr(child, "disable_tp_for_ba_proj", True))
            ):
                continue
            child.sm70_qwen38_fp16_fused_input = True
            fused_gdn_inputs += 1

    if replaced:
        logger.info_once(
            "Prepared %d Qwen3.8 checkpoint-FP16 SM70 M=1 GEMV projections.",
            replaced,
        )
    else:
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in matched the runtime but no "
            "target projections were found."
        )
    if fused_gdn_inputs:
        logger.info_once(
            "Prepared %d Qwen3.8 fused checkpoint-FP16 GDN inputs.",
            fused_gdn_inputs,
        )
    elif envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        logger.warning_once(
            "Qwen3.8 fused checkpoint-FP16 GDN input opt-in found no targets."
        )


__all__ = [
    "Qwen38SM70FP16LinearMethod",
    "_qwen38_fp16_gdn_input_kernel",
    "enable_qwen38_sm70_fp16_gemv",
]
