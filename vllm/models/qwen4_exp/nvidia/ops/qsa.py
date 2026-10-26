# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

# Flash-V100 grouped page4 route (1Cat production attention kernels,
# rebuilt locally against this torch; see /root/run/sm70/PHASE5_QSA_FLASHV100.md).
_QSA_PAGE4_QUERIES = 8
# In-tree grouped-verify head geometry (csrc/attention/qsa_page4_common.h).
_QSA_PAGE4_HEADS = 4
_QSA_PAGE4_OUTPUT_PAGES = 513 * _QSA_PAGE4_QUERIES + 56
_FLASHV100_ENABLE_ENV = "VLLM_SM70_QSA_FLASHV100"
_flashv100_mod = None
_flashv100_attempted = False
_page4_workspaces: dict[tuple[int, int], tuple[object, ...]] = {}

# C-2 M2: TP8 (3-head) page4 route via the M1a 4-head padded reference
# build (kGroupedVerifyHeads 6->4, head 3 zero). Env-gated, default OFF;
# the arm goes live only after the M2 in-service A/B gate (progress.md
# 10-15 晚(11): WIN >= 4800 / DEAD < 4600 tok/s 65K).
#
# Env name deliberately has NO VLLM_ prefix: empirically (10-15 M2 run)
# worker processes in this stack only observed the unprefixed name
# (SM70_QSA_FMA via the sm70-serve systemd drop-in); the VLLM_-prefixed
# variant set in the serve script never reached the workers.
_FLASHV100_TP8_ENABLE_ENV = "SM70_PAGE4_TP8"
# File valve (same class as VLLM_SM70_AR_HIER_FILE): the workers in this
# stack do not observe env set at the API-server/service level, so the
# in-service A/B flips the route by writing a file — no restart.
_FLASHV100_TP8_MODE_FILE = "/tmp/page4_tp8_mode"
_flashv100_tp8_mod = None
_flashv100_tp8_attempted = False
_page4_tp8_workspaces: dict[tuple[int, int], tuple[object, ...]] = {}

# P0 localization (10-16): in-tree stable-ABI QSA page4 backend. Worker
# valve file (same class as _FLASHV100_TP8_MODE_FILE — workers in this
# stack do not see VLLM_-prefixed env): content "local" routes the
# page4/XQA call sites to the torch.ops._C ops built into the vllm
# stable module; missing ops log a warning and keep the .so route.
_QSA_PAGE4_LOCAL_MODE_FILE = "/tmp/qsa_page4_backend"
_qsa_page4_local_checked = False
_qsa_page4_local_mod = None


class _LocalPage4Module:
    """Attribute view of the in-tree page4 torch.ops (_C namespace)."""

    def __getattr__(self, name):
        if not name.startswith(("grouped_sparse_page4", "decode_paged_xqa")):
            raise AttributeError(name)
        return getattr(torch.ops._C, name)


def _qsa_page4_local_ops():
    """Return the in-tree local-backend module when the valve is armed."""
    global _qsa_page4_local_checked, _qsa_page4_local_mod
    if _qsa_page4_local_checked:
        return _qsa_page4_local_mod
    _qsa_page4_local_checked = True
    try:
        with open(_QSA_PAGE4_LOCAL_MODE_FILE) as f:
            mode = f.read().strip()
    except OSError:
        return None
    if mode != "local":
        return None
    if not hasattr(torch.ops._C, "grouped_sparse_page4_plan_fwd"):
        logger.warning(
            "QSA page4 local backend armed but the stable-ABI ops are "
            "absent from the running build; page4 stays off (no .so "
            "fallback exists: the in-tree build is the only source)",
        )
        return None
    _qsa_page4_local_mod = _LocalPage4Module()
    return _qsa_page4_local_mod


def _qsa_page4_tp8_enabled() -> bool:
    """Env first, then the file valve (hot A/B switch, no restart)."""
    if os.getenv(_FLASHV100_TP8_ENABLE_ENV, "0") == "1":
        return True
    return os.path.exists(_FLASHV100_TP8_MODE_FILE)


@lru_cache(maxsize=1)
def _is_sm120() -> bool:
    """True on sm_120 (RTX PRO 6000 Blackwell): selects the sm_120 tuning table."""
    return current_platform.get_device_capability() == (12, 0)


@triton.jit
def _maybe_dequant(tile):
    if tile.dtype == tl.uint8:
        # e4m3 bytes below sm_89 (no Triton fp8e4nv pointer type): sign to bit
        # 15, exp+mantissa to bits 13..7 gives the fp16 of x * 2^-8. Exact for
        # every finite code; the two NaN codes decode to +-480 after the host
        # scale instead of NaN. Decodes four bytes per 32-bit register: the
        # rotate moves each byte's bits 6..1 to 5..0 and its bit 0 to bit 7 of
        # the byte below, lop3 restores the sign bits, prmt pairs high and low
        # bytes, and the mask clears the sign copy at bit 14 and bits 6..0.
        # pack=4 puts element k in byte k of $2 and half k of ($0, $1), the
        # same lane mapping as Triton's own sm<89 fp8 -> fp16 asm.
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 rot, high;
            shf.r.wrap.b32 rot, $2, $2, 1;
            lop3.b32 high, $2, rot, 0x80808080, 0xE4;
            prmt.b32 $0, rot, high, 0x5043;
            prmt.b32 $1, rot, high, 0x7261;
            and.b32 $0, $0, 0xBF80BF80;
            and.b32 $1, $1, 0xBF80BF80;
            }
            """,
            constraints="=r,=r,r",
            args=[tile],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    elif tile.dtype == tl.float8e4nv:
        return tile.to(tl.float16)
    else:
        return tile


@lru_cache(maxsize=1)
def _is_sm80() -> bool:
    """True on sm_80 (A100/A30): selects the wide FP8 byte-decode tiles."""
    return current_platform.get_device_capability() == (8, 0)


@lru_cache(maxsize=1)
def _is_sm90() -> bool:
    """True on sm_90 (H100/H200/H20): selects the sm_90 tuning table."""
    return current_platform.get_device_capability() == (9, 0)


@lru_cache(maxsize=1)
def _is_sm70() -> bool:
    """True on sm_70 (V100): Triton cannot compile fp8e4nv (e4m3) KV here."""
    return current_platform.get_device_capability() == (7, 0)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
    # here for the exp2-based online softmax.
    score_scale = softmax_scale * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
        # measured slower here and less accurate).
        keys = _maybe_dequant(keys).to(query.dtype)
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
        # softmax) so fp16 has the range, its wider mantissa is more
        # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
        values = _maybe_dequant(values)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
    # reciprocal normalizer, so the output is a per-row multiply rather than a
    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
    # so the merge stays correct.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


# SM70 tuning knob: the launch was tuned on GB300 (hundreds of SMs, tensor-core
# paths that tolerate tiny M). Volta needs a deeper software pipeline to hide the
# paged K/V loads, so stages are overridable without editing the call site.
_QSA_NUM_STAGES = envs.VLLM_SM70_QSA_STAGES

# A/B knob for the small-batch (decode) rows of the table: 0 keeps the GB300
# tuned warp counts, N>=8 forces at least N warps per program. The prefill rows
# were a 9x win on Volta; the decode rows are unproven because a synthetic
# random selection gathers far more pages than a real, temporally local one.
_QSA_WARPS_MIN = envs.VLLM_SM70_QSA_WARPS_MIN

def _select_sm120_config(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> tuple[int, int, int]:
    """(block_n, target_splits, num_warps) retuned on sm_120 (RTX PRO 6000
    Blackwell), split by cache dtype: bf16 and fp8 favour different configs on
    sm_120, most visibly on the large-prefill region. Each entry is the fastest
    config that stays correct on its own path; the main win is more warps.
    """
    if is_fp8:
        if base_programs > 2048:
            return (32, 2, 1) if use_prefill_config else (64, 1, 2)
        if base_programs <= 24:
            return 64, 64, 8
        if base_programs <= 32:
            return 128, 8, 4
        if base_programs <= 64:
            return 64, 8, 4
        if base_programs <= 128:
            return 32, 4, 4
        if base_programs <= 256:
            return 128, 4, 4
        if base_programs <= 512:
            return 64, 4, 4
        return 32, 2, 1
    # bf16 K/V cache.
    if base_programs > 2048:
        return 64, 2, 2
    if base_programs <= 24:
        return 64, 64, 8
    if base_programs <= 64:
        return 64, 8, 8
    if base_programs <= 128:
        return 32, 8, 8
    if base_programs <= 256:
        return 32, 8, 1
    if base_programs <= 512:
        return 64, 4, 2
    return 32, 4, 1


def _select_sm90_config(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> tuple[int, int, int]:
    """(block_n, target_splits, num_warps) tuned on sm_90 (H20)."""
    if base_programs > 2048:
        return (32, 1, 1) if use_prefill_config else (64, 1, 2)
    if is_fp8:
        if base_programs <= 24:
            return 64, 64, 2
        if base_programs <= 32:
            return 32, 16, 1
        if base_programs <= 64:
            return 32, 8, 1
        if base_programs <= 128:
            return 32, 4, 1
        if base_programs <= 256:
            return 32, 8, 1
        return 32, 4, 1
    if base_programs <= 32:
        return 64, 64, 2
    if base_programs <= 64:
        return 32, 16, 1
    if base_programs <= 256:
        return 32, 8, 1
    return 32, 4, 1


def _select_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    num_columns: int,
    is_fp8: bool = False,
) -> tuple[int, int, int, int]:
    """Select (block_n, num_warps, num_tiles, num_splits) for the kernel.

    Keyed on base_programs = num_rows * num_kv_heads. The bp > 2048 region splits
    on use_prefill_config (capture-stable: at FULL-graph capture max_query_len is
    the uniform decode/verify length). This default table was tuned on GB300;
    sm_120 (RTX PRO 6000 Blackwell) dispatches to _select_sm120_config instead,
    and sm_90 (Hopper) to _select_sm90_config.
    """
    base_programs = num_rows * num_kv_heads
    if _is_sm120():
        BLOCK_N, target_splits, num_warps = _select_sm120_config(
            base_programs, use_prefill_config, is_fp8
        )
    elif _is_sm90():
        BLOCK_N, target_splits, num_warps = _select_sm90_config(
            base_programs, use_prefill_config, is_fp8
        )
    elif base_programs > 2048:
        if current_platform.is_device_capability(70):
            # Volta: one warp per program holds the whole [BLOCK_M, HEAD_DIM]
            # accumulator, so a prefill launch costs 1259 ms here against 138 ms
            # with 8 warps (rows=4096, 2048-wide selection). BLOCK_N 128 is not
            # even loadable: it wants 160 KB of shared memory vs Volta's 98 KB.
            # 4 warps (promoted 2026-10-13, round-2 A/B): the w8->w4 variant
            # measured 0.509x prefill kernel time (bit-exact) and +25.3%
            # in-service prefill @65K across the round-2 downtime window;
            # the round-2 shadow seam that carried it is removed, value in-place.
            BLOCK_N, target_splits, num_warps = 32, 1, 4
        else:
            BLOCK_N, target_splits, num_warps = (
                (32, 1, 1) if use_prefill_config else (64, 1, 2)
            )
    elif base_programs <= 24:
        BLOCK_N, target_splits, num_warps = 32, 64, 4
    elif base_programs <= 32:
        BLOCK_N, target_splits, num_warps = 32, 16, 1
    elif base_programs <= 64:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 128:
        BLOCK_N, target_splits, num_warps = 32, 4, 1
    elif base_programs <= 256:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 512:
        BLOCK_N, target_splits, num_warps = 64, 4, 2
    else:
        BLOCK_N, target_splits, num_warps = 64, 1, 2
    if num_warps < _QSA_WARPS_MIN:
        num_warps = _QSA_WARPS_MIN
    if is_fp8 and _is_sm80():
        # Wider tiles measured faster for the byte-decode path on A100;
        # unmeasured on sm_86, which keeps the table's tiles.
        BLOCK_N, num_warps = 128, 4
    num_tiles = triton.cdiv(num_columns, BLOCK_N)
    # Never more splits than tiles, never empty.
    num_splits = min(target_splits, num_tiles)
    return BLOCK_N, num_warps, num_tiles, num_splits


def _flashv100_module():
    """In-tree stable-ABI page4 ops: the only page4 source.

    Deliberately no prebuilt .so lookup. The previous version pointed at
    build artifacts whose sources no longer exist, and one of the paths did
    not exist at all, so the route was silently disabled. The in-tree build
    is the 4-head M1a geometry, so the 6-head TP4 path pads each 3-head half
    into it (see _qsa_page4_route).
    """
    global _flashv100_mod, _flashv100_attempted
    if _flashv100_attempted:
        return _flashv100_mod
    _flashv100_attempted = True
    _flashv100_mod = _qsa_page4_local_ops()
    if _flashv100_mod is None:
        logger.error(
            "QSA page4 DISABLED: in-tree stable-ABI ops missing (write "
            "'local' to %s and check `import vllm._custom_ops`); there is no "
            ".so fallback by policy - do not trust page4 numbers",
            _QSA_PAGE4_LOCAL_MODE_FILE,
        )
    return _flashv100_mod


def _qsa_page4_physical_view(cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    """View a [blocks, page, kv, dim] cache as [blocks*page/4, 4, 1, dim]."""
    page = cache.shape[1]
    token_stride = cache.stride(1)
    if cache.stride(0) == page * head_dim:
        return cache.view(cache.shape[0] * (page // 4), 4, 1, head_dim)
    # K/V interleaved layout: the four-token page stride is a narrow
    # re-stride of the token stride.
    return cache.as_strided(
        (cache.shape[0] * (page // 4), 4, 1, head_dim),
        (4 * token_stride, token_stride, token_stride, 1),
    )


def _qsa_page4_nvfp4_pool_view(cache: torch.Tensor) -> torch.Tensor:
    """Validate an nvfp4 pool side view's cell geometry (identity map).

    Pool [P, 2, N, 144] (N tokens; per-token cell [K data 128 | K scale 16 |
    V data 128 | V scale 16]): the side view [P, N, 1, 144] already carries
    the exact strides the in-kernel-dequant readers address (block stride =
    the allocator page, which may be padded past the dense cell content;
    288 token stride; 144 head cell), so no remapping is needed; this only
    fails loud on a geometry mismatch. Microblocks encode 4-token pages:
    N // 4 per pool block (shape-driven, not stride-driven).
    """
    assert cache.dtype == torch.uint8
    n, heads, cell = cache.shape[1], cache.shape[2], cache.shape[3]
    assert heads == 1 and cell == 144 and n % 4 == 0, (
        "nvfp4 QSA pool side view must be [blocks, N, 1, 144] "
        "(cell = data 128 | scale 16), got shape "
        f"{tuple(cache.shape)}"
    )
    s0, s1, s2, s3 = cache.stride()
    assert s3 == 1 and s2 == 144 and s1 == 288 and s0 >= n * 288, (
        "nvfp4 QSA pool side view must have cell strides (S0, 288, 144, 1) "
        f"with S0 >= {n * 288}, got strides {cache.stride()}"
    )
    return cache


_page4_tp4_probe_logged = False
_qsa_entry_probe_count = 0
_qsa_gate_probe_count = 0


def _qsa_page4_eligible(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
    use_prefill_config: bool,
) -> bool:
    """Whether this call can take the Flash-V100 page4 route.

    Pure shape/dtype gates — no GPU reductions (no .item() syncs on the
    hot path). The planner is exact for all visible-window sizes, including
    single-token rows and sparse selections (revalidated 2026-10-01 against
    a token-major reference; an earlier "small-window bug" was a reference
    flattening artifact, see /root/run/sm70/PHASE5_QSA_FLASHV100.md).
    """
    global _page4_tp4_probe_logged
    if not _page4_tp4_probe_logged and q.dim() == 3 and int(q.shape[0]) >= 1024:
        _page4_tp4_probe_logged = True
        logger.info(
            "QSA page4 TP4 probe: q=%s k_cache=%s/%s strides=%s logical=%s "
            "block=%s env=%s prefill_cfg=%s pos=%s seqlen=%s",
            tuple(q.shape), tuple(k_cache.shape), k_cache.dtype,
            (k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
             k_cache.stride(3)),
            tuple(logical_indices.shape), tuple(block_table.shape),
            os.getenv(_FLASHV100_ENABLE_ENV), use_prefill_config,
            query_positions is not None, sequence_lengths is not None)
    if os.getenv(_FLASHV100_ENABLE_ENV, "0") != "1":
        return False
    if not use_prefill_config:
        return False
    if query_positions is None or sequence_lengths is None:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if _flashv100_module() is None:
        return False
    if q.dtype != torch.float16 or k_cache.dtype not in (
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
    ):
        return False
    if q.shape[1] != 6 or q.shape[2] != 256:
        return False
    # Packed selection buffer: selection columns + 1 count column. This
    # model's indexer produces 2051 selection columns (width 2052); the
    # planner ABI is fixed at width 2051 (2050 columns + count), so this
    # gate pins the exact shape the slice below expects.
    if logical_indices.shape[1] != 2052:
        return False
    if k_cache.shape[1] % 4 or k_cache.shape[2] != 1:
        return False
    if block_table.shape[0] != 1:
        # Multi-request batches stay on the Triton route for now.
        return False
    if k_cache.dtype == torch.uint8 and k_cache.shape[3] == 144:
        # nvfp4 cell pool (2026-10-24): the allocator's block stride may be
        # padded past the dense cell content, so the dense-page divisibility
        # check below does not apply. This is TP4's gate: QSA heads ARE
        # TP-sharded, 6 here and 3 at TP8 (measured 2026-10-24), so the
        # 3-head gate above rejects and this one decides the TP4 dispatch.
        if k_cache.stride(1) != 288 or k_cache.stride(2) != 144 or (
            k_cache.stride(3) != 1
        ):
            return False
    # The planner's page-table math needs the block stride to span a whole
    # number of 4-token microblocks (physical_page_stride is a division).
    elif k_cache.stride(0) % (4 * k_cache.stride(1)):
        return False
    # The planner groups rows in 8s; the call length must divide evenly.
    return q.shape[0] % _QSA_PAGE4_QUERIES == 0


_page4_logged = False



def _qsa_verify_page_budget() -> tuple[int, int]:
    """Return (per_query_page_budget, legacy_postplan_token_cap).

    Per-query page budget (new, opt-in): the planner walks each query's
    candidates in descending score order, so a budget of N keeps that query's
    N best 4-token pages and, crucially, stops the hash inserts for the rest.

    ``VLLM_SM70_QSA_VERIFY_PAGE_CAP`` is the legacy knob and keeps its exact
    old behaviour: it is applied AFTER planning to ``out_seq``, i.e. to a
    per-group length over a query-major, POSITION-ordered page list. Despite
    the name it truncates tokens, not pages, and it drops the tail by position
    rather than by score. It is therefore preserved only as a fallback so an
    existing deployment does not change semantics silently; setting
    ``VLLM_SM70_QSA_VERIFY_PAGE_BUDGET`` switches to the score-ordered budget.
    """
    # Hot valve for budget sweeps: a file beats the environment so a single
    # boot can measure several points without a restart. Persisted under
    # /data (not /tmp, which is tmpfs).
    budget = 0
    try:
        with open("/data/sm70_results/qsa_verify_budget") as f:
            budget = int(f.read().strip() or "0")
    except (OSError, ValueError):
        budget = 0
    if budget <= 0:
        budget = int(os.getenv("VLLM_SM70_QSA_VERIFY_PAGE_BUDGET", "0"))
    if budget > 0:
        # Proof-of-effect line (04_standards: prove the change took effect
        # before trusting any number).
        logger.info_once(
            "QSA verify budget ACTIVE: %d pages/query, score-ordered "
            "(planner-bounded, ABI 3)", budget
        )
        return budget, 0
    legacy = int(os.getenv("VLLM_SM70_QSA_VERIFY_PAGE_CAP", "0"))
    if legacy > 0:
        logger.warning(
            "VLLM_SM70_QSA_VERIFY_PAGE_CAP=%d is the legacy post-plan token "
            "cap (position-ordered, not top-N). Keeping its old behaviour; "
            "set VLLM_SM70_QSA_VERIFY_PAGE_BUDGET=<pages/query> (e.g. %d for "
            "the same window) to use the score-ordered planner budget.",
            legacy,
            max(1, legacy // (4 * _QSA_PAGE4_QUERIES)),
        )
        return 0, legacy
    return 0, 0


def _qsa_page4_route(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Run the Flash-V100 grouped page4 planner + fwd into out."""
    global _page4_audit_done, _page4_logged
    if not _page4_logged:
        _page4_logged = True
        logger.info(
            "QSA page4 route active: rows=%d heads=%d dim=%d (Flash-V100)",
            q.shape[0], q.shape[1], q.shape[2],
        )
    m = _flashv100_module()
    rows = q.shape[0]
    groups = rows // _QSA_PAGE4_QUERIES
    device_index = q.device.index if q.device.index is not None else -1
    key = (
        device_index,
        int(torch.cuda.current_stream(q.device).cuda_stream),
    )
    ws = _page4_workspaces.get(key)
    if ws is None or ws[0] < groups:
        capacity = 1 << (groups - 1).bit_length()
        # The planner only writes the first out_seq[group] entries of each
        # 4160-wide page row; the fwd's tile loop still READS the stale tail
        # (tokens are masked out, but their V data is multiplied in). A
        # stale id of 0 reads a real, finite microblock, so the masked
        # 0.0 * finite product stays finite; torch.empty garbage ids can
        # point at NaN bit patterns and 0.0 * NaN = NaN poisons a single
        # output dim (observed in the 2026-10-02 in-engine audit).
        ws = (
            capacity,
            torch.zeros(
                (capacity, _QSA_PAGE4_OUTPUT_PAGES),
                dtype=torch.int32,
                device=q.device,
            ),
            torch.zeros(
                (capacity, _QSA_PAGE4_OUTPUT_PAGES),
                dtype=torch.uint32,
                device=q.device,
            ),
            torch.zeros(
                (capacity,), dtype=torch.int32, device=q.device
            ),
            torch.empty(
                (capacity * _QSA_PAGE4_QUERIES, _QSA_PAGE4_HEADS),
                dtype=torch.float32,
                device=q.device,
            ),
            torch.empty(
                (capacity * _QSA_PAGE4_QUERIES, _QSA_PAGE4_HEADS, q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            ),
            torch.empty(
                (capacity * _QSA_PAGE4_QUERIES, _QSA_PAGE4_HEADS, q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            ),
        )
        _page4_workspaces[key] = ws
    (
        _,
        _cap_pages,
        _cap_masks,
        _cap_seq,
        _cap_lse,
        _cap_qp4,
        _cap_outp4,
    ) = ws
    # The C++ planner/fwd require the metadata buffers to have EXACTLY
    # [groups, ...] rows (exact-match TORCH_CHECK). Pass row views of the
    # capacity-sized workspace so the per-stream allocation reuse still
    # works when a later call has fewer groups than the buffer's capacity
    # (mixed chunk sizes under concurrent requests — the R=4 soak crash
    # of 2026-10-02: a 1024-group prefill buffer reused by a 256-group
    # call failed the planner's exact-row check and killed the engine).
    out_pages = _cap_pages[:groups]
    out_masks = _cap_masks[:groups]
    out_seq = _cap_seq[:groups]
    lse = _cap_lse[:rows]
    q_p4 = _cap_qp4[:rows]
    out_p4 = _cap_outp4[:rows]

    # Planner ABI is fixed at [8N, 2051]: 2050 index columns + count. Our
    # buffer is one index column wider, so hand the planner the first 2050
    # columns with the count clamped to what they hold — bit-identical to
    # 1Cat's production buffer (their indexer fits 2050 columns).
    if os.getenv("VLLM_SM70_QSA_PLAN_PACK", "0") == "1":
        # Candidate I: one launch instead of clamp + cat, same integers
        # (gated, default 0 = byte-identical main path; the module is only
        # imported when the valve is on).
        from .qsa_plan_pack import qsa_plan_pack

        idx_p4 = qsa_plan_pack(logical_indices)
    else:
        idx_p4 = torch.cat(
            [
                logical_indices[:, :2050],
                logical_indices[:, 2051:2052].clamp(max=2050),
            ],
            dim=1,
        )
    # Score-ordered budget (new, opt-in) vs the legacy post-plan token cap.
    _page_budget, _legacy_cap = _qsa_verify_page_budget()
    m.grouped_sparse_page4_plan_fwd(
        idx_p4,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out_pages,
        out_masks,
        out_seq,
        k_cache.shape[1],
        # Microblocks per cache block in the page4 view's index space.
        # MUST use the token stride (stride(1)), not the head dim: the
        # engine's K/V are interleaved in the token slot (token stride
        # 512 = 2*dim here), so 4*dim underestimates the 4-token page span
        # and the planner's page table points at wrong microblocks.
        k_cache.shape[1] // 4
        if k_cache.dtype == torch.uint8
        else k_cache.stride(0) // (4 * k_cache.stride(1)),
        k_cache.shape[0],
        _page_budget,
    )
    if _legacy_cap > 0:
        out_seq.clamp_(max=_legacy_cap)
    kv_dtype = {
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.float8_e5m2: "fp8_e5m2",
        torch.uint8: "nvfp4",
    }.get(k_cache.dtype, "auto")
    if kv_dtype != "auto":
        k_cache = k_cache.view(torch.uint8)
        v_cache = v_cache.view(torch.uint8)
    if kv_dtype == "nvfp4":
        # The nvfp4 reader walks cell-interleaved pool blocks; the
        # token-strided physical-view helper does not apply (same split as
        # the 3-head route).
        k_phys = _qsa_page4_nvfp4_pool_view(k_cache)
        v_phys = _qsa_page4_nvfp4_pool_view(v_cache)
    else:
        k_phys = _qsa_page4_physical_view(k_cache, q.shape[2])
        v_phys = _qsa_page4_physical_view(v_cache, q.shape[2])
    # The in-tree build is the 4-head M1a geometry (kGroupedVerifyHeads == 4)
    # while TP4 has 6 QSA heads. The planner output (pages/masks/seq_lens)
    # depends only on the token rows and the index tables, not on heads, so
    # one plan serves two passes that each carry 3 real heads padded with a
    # zero head. This keeps a single kernel source in the tree: no prebuilt
    # artifact, no second head geometry to maintain.
    abi2 = int(getattr(m, "grouped_sparse_page4_abi_version", lambda: 1)()) >= 2
    for head_start in range(0, q.shape[1], _QSA_PAGE4_HEADS - 1):
        head_stop = min(head_start + _QSA_PAGE4_HEADS - 1, q.shape[1])
        real = head_stop - head_start
        q_p4[:, :real].copy_(q[:, head_start:head_stop])
        q_p4[:, real:].zero_()
        args = (
            q_p4,
            k_phys,
            v_phys,
            out_p4,
            out_pages,
            out_masks,
            out_seq,
            lse,
            q.shape[2] ** -0.5,
        )
        if abi2:
            m.grouped_sparse_page4_fwd(
                *args, _abl_kv_dtype(kv_dtype), k_scale, v_scale
            )
        else:
            m.grouped_sparse_page4_fwd(*args)
        out[:, head_start:head_stop].copy_(out_p4[:, :real])
    if os.getenv("VLLM_SM70_PAGE4_AUDIT", "0") == "1" and not _page4_audit_done:
        _page4_audit_done = True
        _qsa_page4_audit(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req,
            query_positions, sequence_lengths, out, output_gate_view,
            out_pages, out_masks, out_seq, lse,
        )
    # Same gate semantics as the Triton kernels: the kernel loads the full
    # dim vector output_gate[row, head, dim] (a [G, D] load) and multiplies
    # elementwise — the gate is per-(row, head, dim), not a per-head scalar.
    # (The model's gate projection has width heads*(1+gate), i.e. the gate
    # carries one value per output dim.)
    out.mul_(torch.sigmoid(output_gate_view))


def _flashv100_tp8_module():
    """In-tree stable-ABI page4 ops (4-head M1a geometry).

    Same single-source rule as _flashv100_module: no artifact lookup.
    """
    global _flashv100_tp8_mod, _flashv100_tp8_attempted
    if _flashv100_tp8_attempted:
        return _flashv100_tp8_mod
    _flashv100_tp8_attempted = True
    _flashv100_tp8_mod = _qsa_page4_local_ops()
    if _flashv100_tp8_mod is None:
        logger.error(
            "QSA page4 TP8 DISABLED: in-tree stable-ABI ops missing (write "
            "'local' to %s and check `import vllm._custom_ops`); there is no "
            ".so fallback by policy - do not trust page4 numbers",
            _QSA_PAGE4_LOCAL_MODE_FILE,
        )
    return _flashv100_tp8_mod


def _qsa_page4_tp8_eligible(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
    use_prefill_config: bool,
) -> bool:
    """TP8 3-head page4 gate (same shape rules as the TP4 gate, heads==3)."""
    if not _qsa_page4_tp8_enabled():
        return False
    if not use_prefill_config:
        return False
    if query_positions is None or sequence_lengths is None:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if _flashv100_tp8_module() is None:
        return False
    if q.dtype != torch.float16 or k_cache.dtype not in (
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
    ):
        return False
    if q.shape[1] != 3 or q.shape[2] != 256:
        return False
    if logical_indices.shape[1] != 2052:
        return False
    if k_cache.shape[1] % 4 or k_cache.shape[2] != 1:
        return False
    if block_table.shape[0] != 1:
        return False
    if k_cache.dtype == torch.uint8 and k_cache.shape[3] == 144:
        # nvfp4 cell pool: the allocator's block stride may be padded past
        # the dense cell content, so the dense-page divisibility check does
        # not apply; the cell geometry itself is gated by the pool view.
        # The block stride spans one cell (dense 4 GB TP4 pool) or two
        # (padded 9.6 GB TP8 pool); the kernel receives the strides
        # explicitly, so both are valid and the relation is checked against
        # the cell width instead of a hardcoded 288.
        if k_cache.stride(1) not in (
            k_cache.shape[3],
            2 * k_cache.shape[3],
        ) or k_cache.stride(2) != k_cache.shape[3] or k_cache.stride(3) != 1:
            return False
    elif k_cache.stride(0) % (4 * k_cache.stride(1)):
        return False
    return q.shape[0] % _QSA_PAGE4_QUERIES == 0


_page4_tp8_logged = False


def _abl_kv_dtype(kv_dtype: str) -> str:
    """Arm the in-tree C++ ABL channel (kv_cache_dtype '@N' suffix). File valve,
    read per call; absent => unchanged (default off)."""
    try:
        with open("/tmp/sm70_page4_abl_local") as f:
            n = f.read().strip()
    except OSError:
        return kv_dtype
    return f"{kv_dtype}@{int(n)}" if n else kv_dtype


def _qsa_page4_tp8_route(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """TP8 page4: pad q to the M1a 4-head geometry (head 3 zero), run
    planner + fwd, copy the 3 real heads into out, apply the gate."""
    global _page4_tp8_logged
    if not _page4_tp8_logged:
        _page4_tp8_logged = True
        logger.info(
            "QSA page4 TP8 route active: rows=%d heads=%d dim=%d (M1a)",
            q.shape[0], q.shape[1], q.shape[2],
        )
    m = _flashv100_tp8_module()
    rows = q.shape[0]
    groups = rows // _QSA_PAGE4_QUERIES
    device_index = q.device.index if q.device.index is not None else -1
    key = (
        device_index,
        int(torch.cuda.current_stream(q.device).cuda_stream),
    )
    ws = _page4_tp8_workspaces.get(key)
    if ws is None or ws[0] < groups:
        capacity = 1 << (groups - 1).bit_length()
        ws = (
            capacity,
            torch.zeros(
                (capacity, _QSA_PAGE4_OUTPUT_PAGES),
                dtype=torch.int32, device=q.device,
            ),
            torch.zeros(
                (capacity, _QSA_PAGE4_OUTPUT_PAGES),
                dtype=torch.uint32, device=q.device,
            ),
            torch.zeros(
                (capacity,), dtype=torch.int32, device=q.device
            ),
            torch.empty(
                (capacity * _QSA_PAGE4_QUERIES, 4),
                dtype=torch.float32, device=q.device,
            ),
            torch.empty(
                (capacity * _QSA_PAGE4_QUERIES, 4, 256),
                dtype=torch.float16, device=q.device,
            ),
            torch.empty(
                (capacity * _QSA_PAGE4_QUERIES, 4, 256),
                dtype=torch.float16, device=q.device,
            ),
        )
        _page4_tp8_workspaces[key] = ws
    _cap, cap_pages, cap_masks, cap_seq, cap_lse, cap_qp4, cap_outp4 = ws
    out_pages = cap_pages[:groups]
    out_masks = cap_masks[:groups]
    out_seq = cap_seq[:groups]
    lse = cap_lse[:rows]
    q_p4 = cap_qp4[:rows]
    out_p4 = cap_outp4[:rows]
    q_p4[:, :3].copy_(q)
    q_p4[:, 3].zero_()
    if os.getenv("VLLM_SM70_QSA_PLAN_PACK", "0") == "1":
        # Candidate I: one launch instead of clamp + cat, same integers
        # (gated, default 0 = byte-identical main path; the module is only
        # imported when the valve is on).
        from .qsa_plan_pack import qsa_plan_pack

        idx_p4 = qsa_plan_pack(logical_indices)
    else:
        idx_p4 = torch.cat(
            [
                logical_indices[:, :2050],
                logical_indices[:, 2051:2052].clamp(max=2050),
            ],
            dim=1,
        )
    # Score-ordered budget (new, opt-in) vs the legacy post-plan token cap.
    _page_budget, _legacy_cap = _qsa_verify_page_budget()
    m.grouped_sparse_page4_plan_fwd(
        idx_p4,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out_pages,
        out_masks,
        out_seq,
        k_cache.shape[1],
        # The nvfp4 cell pool's block stride is the allocator page (possibly
        # padded past the dense cells), so the 4-token pages-per-block must
        # come from the shape, not the stride ratio.
        k_cache.shape[1] // 4
        if k_cache.dtype == torch.uint8
        else k_cache.stride(0) // (4 * k_cache.stride(1)),
        k_cache.shape[0],
        _page_budget,
    )
    if _legacy_cap > 0:
        out_seq.clamp_(max=_legacy_cap)
    # C-3: the reference kernel reads the packed fp8 KV in-kernel
    # (e4m3/e5m2 bytes + per-tensor scales); fp16 stays "auto".
    kv_dtype = {
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.float8_e5m2: "fp8_e5m2",
        torch.uint8: "nvfp4",
    }.get(k_cache.dtype, "auto")
    # The reference kernel decodes fp8 bytes in-kernel and hard-checks the
    # storage dtype: fp8 KV must arrive as raw uint8, not an fp8 view.
    if kv_dtype != "auto":
        k_cache = k_cache.view(torch.uint8)
        v_cache = v_cache.view(torch.uint8)
    if kv_dtype == "nvfp4":
        # The nvfp4 reader walks cell-interleaved pool blocks (k side at the
        # block start, v side at +144); the physical-view helper assumes an
        # fp16/fp8 token-strided layout and does not apply. The pool view
        # validates the cell geometry and passes the view through.
        k_phys = _qsa_page4_nvfp4_pool_view(k_cache)
        v_phys = _qsa_page4_nvfp4_pool_view(v_cache)
    else:
        k_phys = _qsa_page4_physical_view(k_cache, 256)
        v_phys = _qsa_page4_physical_view(v_cache, 256)
    m.grouped_sparse_page4_fwd(
        q_p4,
        k_phys,
        v_phys,
        out_p4,
        out_pages,
        out_masks,
        out_seq,
        lse,
        256 ** -0.5,
        _abl_kv_dtype(kv_dtype),
        k_scale,
        v_scale,
    )
    out.copy_(out_p4[:, :3])
    out.mul_(torch.sigmoid(output_gate_view))
# XQA page4 batch route: FP8-KV decode and unaligned prefill rows through
# Flash-V100's paged XQA kernel (1Cat production decode path). Ported from
# 1cat-vllm ops/qsa.py; the geometry constants are the same 2051-selection
# page4 layout the grouped planner uses.
_XQA_PAGE4_PARTITION = 1024
_XQA_PAGE4_PAGES = 513
_XQA_PAGE4_MARKER = 1 << 30
_xqa_page4_workspaces: dict[
    tuple[int, int, int, int, int, bool], tuple[object, ...]
] = {}
_xqa_table_precompiled = False
_xqa_route_logged = False


def _qsa_xqa_padded_heads(heads: int, kv_dtype: str) -> int:
    """XQA q_per_kv contract: the e4m3 kernel requires q_per_kv=6; the
    other dtypes accept {4, 6, 8}. TP8's 3 q heads pad with zero heads."""
    if kv_dtype == "fp8_e4m3":
        return 6
    if heads in (4, 6, 8):
        return heads
    return 1 << (heads - 1).bit_length()


@triton.jit(
    do_not_specialize=[
        "stride_indices_row",
        "stride_table_req",
        "stride_encoded_row",
        "rows",
        "num_cache_blocks",
        "num_requests",
    ]
)
def _qsa_xqa_page4_table_kernel(
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    encoded_pages_ptr,
    xqa_sequence_lengths_ptr,
    stride_indices_row,
    stride_table_req,
    stride_encoded_row,
    rows,
    num_cache_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    COMPLETE_PAGES: tl.constexpr,
    OUTPUT_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    PHYSICAL_PAGE_STRIDE: tl.constexpr,
    SORT_INLINE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_PAGES)
    request = tl.load(token_to_req_ptr + row)
    request_is_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=request_is_valid,
        other=0,
    )
    # Selection loop bound from the count column (last column of the packed
    # selection buffer: 2050 index columns + count, indexer ABI). The indexer
    # writes count = 0 for inert rows; a row whose query/sequence metadata
    # claims a live sequence but whose selection is empty or stale must
    # degrade to xseq = 0, not to garbage pages (an all-garbage vbt row with
    # a large xseq makes the XQA reader walk i64max-sorted entries = IMA,
    # EXP-12f 2026-10-16). For active rows the indexer's count equals the
    # visible-token formula below, so this clamp is a no-op there.
    sel_count = tl.load(
        indices_ptr + row * stride_indices_row + (stride_indices_row - 1),
        mask=row < rows,
        other=0,
    )
    sel_count = tl.minimum(tl.maximum(sel_count, 0), stride_indices_row - 1)
    # Padded graph rows use position -1. Clamp malformed or stale positions to
    # the request's live sequence so they cannot expose a synthetic tail page.
    visible_tokens = tl.minimum(
        tl.maximum(query_position + 1, 0),
        sequence_length,
    )
    complete_pages = tl.minimum(
        tl.minimum(visible_tokens // 4, sequence_length // 4),
        COMPLETE_PAGES,
    )
    tail_count = visible_tokens - (visible_tokens // 4) * 4
    is_complete = slots < complete_pages
    selected_token = tl.load(
        indices_ptr + row * stride_indices_row + slots * 4,
        mask=(row < rows) & is_complete,
        other=-1,
    )
    tail_token = (visible_tokens // 4) * 4
    selected_tail_token = tl.load(
        indices_ptr + row * stride_indices_row + complete_pages * 4,
        mask=(row < rows) & (tail_count > 0),
        other=-1,
    )
    tail_is_valid = (
        (tail_count > 0)
        & (selected_tail_token == tail_token)
        & (selected_tail_token < sequence_length)
    )
    is_tail = (slots == complete_pages) & tail_is_valid
    logical_token = tl.where(is_tail, selected_tail_token, selected_token)
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token - logical_page * PAGE_SIZE
    valid = (
        (row < rows)
        & request_is_valid
        & (logical_token >= 0)
        & (logical_token < sequence_length)
        & (logical_page < PAGE_TABLE_WIDTH)
        & (is_complete | is_tail)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    physical_microblock = (
        tl.maximum(physical_page, 0) * PHYSICAL_PAGE_STRIDE + page_offset // 4
    )
    # Sort by logical token, not allocator-dependent physical page ID. Keep
    # the partial causal page after all complete pages and invalid slots last.
    logical_key = safe_token.to(tl.int64) << 31
    encoded = tl.where(
        valid & is_complete,
        logical_key | physical_microblock.to(tl.int64),
        tl.where(
            valid & is_tail,
            (1 << 62) | logical_key | physical_microblock.to(tl.int64),
            9223372036854775807,
        ),
    )
    # Canonical page order without a second launch: the indexer emits the
    # selected tokens in score order (racy on near-ties), so the encoded values
    # are NOT sorted; sorting here reproduces torch.sort's order deterministically
    # (invalid sentinels sort last, the tail page after all complete pages).
    # SORT_INLINE: 0 = leave to torch.sort, 1 = sort the int64 encoding,
    # 2 = sort a 32-bit token key and recompute the microblock (half the bytes
    # per compare-exchange; tokens are unique so token order fixes page order).
    if SORT_INLINE == 1:
        encoded = tl.sort(encoded)
    elif SORT_INLINE == 2:
        key = tl.where(
            valid & is_tail,
            (1 << 30) + safe_token,
            tl.where(valid & is_complete, safe_token, 2147483647),
        )
        key = tl.sort(key)
        sorted_invalid = key == 2147483647
        tok = key & ((1 << 30) - 1)
        page = tok // PAGE_SIZE
        phys = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(page, PAGE_TABLE_WIDTH - 1),
            mask=(~sorted_invalid) & (row < rows),
            other=-1,
        )
        micro = tl.maximum(phys, 0) * PHYSICAL_PAGE_STRIDE + (
            tok - page * PAGE_SIZE
        ) // 4
        encoded = tl.where(
            sorted_invalid, 9223372036854775807, micro.to(tl.int64)
        )
    tl.store(
        encoded_pages_ptr + row * stride_encoded_row + slots,
        encoded,
        mask=(row < rows) & (slots < OUTPUT_PAGES),
    )
    tl.store(
        xqa_sequence_lengths_ptr + row,
        tl.minimum(
            complete_pages * 4 + tl.where(tail_is_valid, tail_count, 0),
            sel_count,
        ),
        mask=row < rows,
    )


def _qsa_xqa_page4_block_table(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    num_cache_blocks: int,
    page_size: int,
    physical_page_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if physical_page_stride is None:
        physical_page_stride = page_size // 4
    rows = logical_indices.shape[0]
    encoded_pages = torch.empty(
        (rows, _XQA_PAGE4_PAGES),
        dtype=torch.int64,
        device=logical_indices.device,
    )
    xqa_sequence_lengths = torch.empty(
        (rows,), dtype=torch.int32, device=logical_indices.device
    )
    _page4_sort_mode = envs.VLLM_SM70_QSA_PAGE4_SORT
    _qsa_xqa_page4_table_kernel[(rows,)](
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        encoded_pages,
        xqa_sequence_lengths,
        logical_indices.stride(0),
        block_table.stride(0),
        encoded_pages.stride(0),
        rows,
        num_cache_blocks,
        block_table.shape[0],
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        COMPLETE_PAGES=2048 // 4,
        OUTPUT_PAGES=_XQA_PAGE4_PAGES,
        BLOCK_PAGES=1024,
        PHYSICAL_PAGE_STRIDE=physical_page_stride,
        SORT_INLINE=(
            2 if _page4_sort_mode == "inline32"
            else 1 if _page4_sort_mode == "inline"
            else 0
        ),
        num_warps=16,
    )
    # "inline"  - the kernel sorted the encoded values (tl.sort): canonical order,
    #             no second launch.
    # "skip"    - trust the emit order (measured NON-deterministic: the indexer's
    #             selection is racy on near-ties, so the page order varies per run).
    # "torch"   - baseline: torch.sort (12 radixSort launches/step).
    if _page4_sort_mode in ("skip", "inline", "inline32"):
        pages = encoded_pages
    else:
        pages = torch.sort(encoded_pages, dim=1).values
    physical_pages = torch.bitwise_and(
        pages,
        _XQA_PAGE4_MARKER - 1,
    ).to(torch.int32)
    return physical_pages, xqa_sequence_lengths


def _qsa_xqa_page4_workspace(
    q: torch.Tensor,
    num_partitions: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """XQA workspaces keyed by (device, stream, padded heads, dim,
    partitions, e4m3): partial outputs, merge stats, partition count,
    and the head-padded q/out buffers (the kernel wants q_per_kv in
    {4, 6, 8}; TP8's 3 q heads pad with a zero head)."""
    device_index = q.device.index if q.device.index is not None else -1
    padded_heads = _qsa_xqa_padded_heads(q.shape[1], kv_cache_dtype)
    key = (
        device_index,
        int(torch.cuda.current_stream(q.device).cuda_stream),
        padded_heads,
        q.shape[2],
        num_partitions,
        kv_cache_dtype == "fp8_e4m3",
    )
    ws = _xqa_page4_workspaces.get(key)
    rows = q.shape[0]
    if ws is None or ws[0] < rows:
        # torch.full (scalar kernel arg), never torch.tensor([...],
        # device=cuda) — the latter is a host copy, illegal under capture.
        capacity = 1 << (rows - 1).bit_length()
        ws = (
            capacity,
            torch.empty(
                (capacity, padded_heads, num_partitions, q.shape[2]),
                dtype=(
                    torch.float32
                    if kv_cache_dtype == "fp8_e4m3"
                    else torch.float16
                ),
                device=q.device,
            ),
            torch.empty(
                (capacity, padded_heads, num_partitions),
                dtype=torch.float32,
                device=q.device,
            ),
            torch.empty(
                (capacity, padded_heads, num_partitions),
                dtype=torch.float32,
                device=q.device,
            ),
            torch.full(
                (1,), num_partitions, dtype=torch.int32, device=q.device
            ),
            torch.empty(
                (capacity, padded_heads, q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            ),
            torch.empty(
                (capacity, padded_heads, q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            ),
        )
        _xqa_page4_workspaces[key] = ws
    (
        _cap,
        temporary_output,
        max_logits,
        exp_sums,
        active_num_partitions,
        q_pad,
        out_pad,
    ) = ws
    return (
        temporary_output[:rows],
        max_logits[:rows],
        exp_sums[:rows],
        active_num_partitions,
        q_pad[:rows],
        out_pad[:rows],
    )


def _qsa_xqa_ensure_table_compiled(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> None:
    """Compile the XQA table kernel eagerly (first FP8 call, which is the
    warmup prefill) so it never JITs inside a CUDA graph capture."""
    global _xqa_table_precompiled
    if (
        _xqa_table_precompiled
        or query_positions is None
        or sequence_lengths is None
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    _qsa_xqa_page4_block_table(
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        k_cache.shape[0],
        k_cache.shape[1],
        k_cache.shape[1] // 4
        if k_cache.dtype == torch.uint8
        else k_cache.stride(0) // (4 * k_cache.stride(1)),
    )
    _xqa_table_precompiled = True


def _qsa_xqa_page4_eligible(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    """FP8-KV rows (decode or unaligned prefill) take the XQA batch route."""
    if not _qsa_page4_tp8_enabled():
        return False
    if k_cache.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
    ):
        return False
    if k_cache.dtype == torch.float8_e4m3fn and (
        logical_indices.shape[0] > 1
        and os.environ.get("VLLM_FLASH_V100_E4M3_BATCH_XQA", "1") != "1"
    ):
        # e4m3 XQA runs the P256 path on the virtual page4 view; B>=2
        # needs the .so batch env, which defaults to "1" (unset = on,
        # matching the 1cat legacy_policy default).
        return False
    # Local q heads = num_attention_heads / TP. For this model (24 q heads,
    # 8 GPUs) the feasible TP is {4, 8} -> 6 or 3 local heads. The wide
    # nvfp4 TC kernel takes q_per_kv in {4, 6, 8}; the workspace pads 3 -> 4
    # with a zero head and passes 6/8 through. (TP must divide the GPU count
    # and the head counts; odd TP does not apply here.)
    if q.dtype != torch.float16 or q.shape[1] not in (3, 4, 6, 8) or q.shape[2] != 256:
        return False
    if query_positions is None or sequence_lengths is None:
        return False
    # The Triton table kernel must already be compiled when capturing.
    if (
        torch.cuda.is_current_stream_capturing()
        and not _xqa_table_precompiled
    ):
        return False
    if logical_indices.shape[1] != 2052:
        return False
    if k_cache.shape[1] % 4 or k_cache.shape[2] != 1:
        return False
    if k_cache.dtype == torch.uint8 and k_cache.shape[3] == 144:
        # nvfp4 cell pool: allocator block stride may be padded past the
        # dense cells, so the dense-page divisibility check does not apply;
        # the cell geometry itself is gated by the pool view.
        if k_cache.stride(1) != 288 or k_cache.stride(2) != 144 or (
            k_cache.stride(3) != 1
        ):
            return False
    elif k_cache.stride(0) % (4 * k_cache.stride(1)):
        return False
    m = _flashv100_tp8_module()
    return m is not None and hasattr(m, "decode_paged_xqa_fwd")


_xqa_audit_env = os.environ.get("VLLM_SM70_XQA_AUDIT", "0") == "1"


def _xqa_audit_vbt(
    vbt: torch.Tensor,
    xseq: torch.Tensor,
    total_microblocks: int,
    li_r: torch.Tensor,
    block_table: torch.Tensor,
    ttr_r: torch.Tensor,
    qpos_r: torch.Tensor,
    seq_lens_r: torch.Tensor,
    desc: str,
) -> None:
    """Observation-only guard: audit the vbt slots the kernel will read
    (the per-row xseq window). On an out-of-range value, dump the per-row
    metadata so the journal captures the crash-time inputs; the XQA launch
    proceeds unchanged (no behavior change). Skipped while a CUDA graph is
    captured (host sync is illegal there).
    """
    if torch.cuda.is_current_stream_capturing():
        return
    rows = vbt.shape[0]
    width = vbt.shape[1]
    npages = (xseq + 3) // 4
    xseq_bad = (xseq < 0) | (xseq > width * 4) | (npages > width)
    cols = torch.arange(width, device=vbt.device, dtype=torch.int64)
    valid = cols[None, :] < npages.to(torch.int64)[:, None]
    vals = vbt.to(torch.int64)
    bad_val = (vals >= total_microblocks) & valid
    if bool(xseq_bad.any().item()) or bool(bad_val.any().item()):
        why = "xseq out of range" if bool(xseq_bad.any().item()) else "vbt OOB"
        logger.error(
            "XQA vbt audit BAD (%s): %s xseq=%s total_microblocks=%d",
            why, desc, xseq.tolist(), total_microblocks,
        )
        for r in range(rows):
            bad_slots = bad_val[r].nonzero().flatten().tolist()[:8]
            logger.error(
                "  row %d: xseq=%d xseq_bad=%s bad_slots=%s bad_vals=%s "
                "li[:8]=%s count=%d bt[:8]=%s t2r=%d qp=%d seq_len=%d",
                r, int(xseq[r]), bool(xseq_bad[r].item()), bad_slots,
                [int(vbt[r, s]) for s in bad_slots],
                li_r[r, :8].tolist(), int(li_r[r, -1]),
                block_table[ttr_r[r], :8].tolist(), int(ttr_r[r]),
                int(qpos_r[r]), int(seq_lens_r[r]),
            )
        return
    # Per-call line (not one-shot): under CUDA_LAUNCH_BLOCKING the last
    # line per rank before the crash IS the faulting call, with its full
    # per-row inputs.
    row_max = torch.where(valid, vals, -1).max(dim=1).values.tolist()
    logger.info(
        "XQA vbt audit ok: %s xseq=%s vbt_row_max=%s "
        "total_microblocks=%d",
        desc, xseq.tolist(), row_max, total_microblocks,
    )


def _qsa_xqa_page4_route(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Run the Flash-V100 paged XQA batch into out, then apply the gate."""
    global _xqa_route_logged
    if not _xqa_route_logged:
        _xqa_route_logged = True
        logger.info(
            "QSA XQA page4 route active: rows=%d heads=%d dim=%d",
            q.shape[0], q.shape[1], q.shape[2],
        )
    m = _flashv100_tp8_module()
    head_dim = q.shape[2]
    kv_dtype = {
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.float8_e5m2: "fp8_e5m2",
        torch.uint8: "nvfp4",
    }.get(k_cache.dtype, "auto")
    # Raw-byte contract, same as the TP8 route: the reference kernel decodes
    # fp8 in-kernel and hard-checks uint8 storage.
    k_raw = k_cache.view(torch.uint8)
    v_raw = v_cache.view(torch.uint8)
    # The e4m3 XQA kernel only supports P256 for virtual page4 caches; the
    # nvfp4 reader takes the same generic tc-256-wide route (its G6 dual-CTA
    # specialization is fp16/fp8-only, see qsa_page4_xqa.cu).
    # The partition kernel's grid is (batch, kv_heads, partitions) and its
    # merge cost scales with the partition count. Measured at 65K/B=1 (salt
    # 9503, all sizes instantiated in DISPATCH_PARTITION): 3 partitions 74.6,
    # 5 -> 76.4, 9 -> 77.0, 17 -> 76.8, 33 -> 75.5 tok/s. The optimum keeps
    # ~8K tokens (64 blocks) of KV per partition, which is exactly the value
    # below at every sequence length; longer sequences get more partitions,
    # shorter ones fewer, so neither the batch nor the length needs a
    # separate knob.
    partition_size = (
        64 if kv_dtype == "nvfp4"
        else 256 if kv_dtype == "fp8_e4m3"
        else _XQA_PAGE4_PARTITION
    )
    num_partitions = math.ceil(logical_indices.shape[1] / partition_size)
    # Microblocks per cache block in the physical view's index space.
    # nvfp4 cell pool: shape-driven (the allocator block stride may be
    # padded, so the stride ratio would under/over-count).
    page_stride = (
        k_raw.shape[1] // 4
        if kv_dtype == "nvfp4"
        else k_raw.stride(0) // (4 * k_raw.stride(1))
    )
    if kv_dtype == "nvfp4":
        # Cell pool views: the nvfp4 reader addresses [data 128 | scale 16]
        # cells (288 B per token, 144 B per side) with a runtime
        # pages-per-block from the view shape.
        k_phys = _qsa_page4_nvfp4_pool_view(k_raw)
        v_phys = _qsa_page4_nvfp4_pool_view(v_raw)
    else:
        k_phys = _qsa_page4_physical_view(k_raw, head_dim)
        v_phys = _qsa_page4_physical_view(v_raw, head_dim)

    def _run(
        q_r: torch.Tensor,
        li_r: torch.Tensor,
        ttr_r: torch.Tensor,
        qpos_r: torch.Tensor,
        out_r: torch.Tensor,
    ) -> None:
        virtual_block_table, xqa_sequence_lengths = (
            _qsa_xqa_page4_block_table(
                li_r,
                block_table,
                ttr_r,
                qpos_r,
                sequence_lengths,
                k_cache.shape[0],
                k_cache.shape[1],
                page_stride,
            )
        )
        if _xqa_audit_env:
            _xqa_audit_vbt(
                virtual_block_table, xqa_sequence_lengths,
                k_cache.shape[0] * page_stride,
                li_r, block_table, ttr_r, qpos_r, sequence_lengths,
                f"kv={kv_dtype} rows={q_r.shape[0]}",
            )
        temporary_output, max_logits, exp_sums, active_parts, q_pad, out_pad = (
            _qsa_xqa_page4_workspace(q_r, num_partitions, kv_dtype)
        )
        if q_pad.shape[1] == q_r.shape[1]:
            qk, outk = q_r, out_r
        else:
            # 3 q heads: pad with a zero head for the kernel, copy the
            # real heads back after. The pad heads MUST be zeroed on every
            # call: q_pad is a reused torch.empty workspace, so stale /
            # uninitialized fp16 NaN bit patterns in the pad head poison
            # the q_per_kv group reduction and NaN out the real heads too
            # (same reason the M1a page4 route does q_p4[:, 3].zero_()).
            qk, outk = q_pad, out_pad
            qk[:, : q_r.shape[1]].copy_(q_r)
            qk[:, q_r.shape[1] :].zero_()
        m.decode_paged_xqa_fwd(
            qk,
            k_phys,
            v_phys,
            outk,
            virtual_block_table,
            xqa_sequence_lengths,
            temporary_output,
            max_logits,
            exp_sums,
            active_parts,
            head_dim**-0.5,
            partition_size,
            num_partitions,
            kv_dtype,
            k_scale,
            v_scale,
            -1,
            -1,
            0,
        )
        if outk is not out_r:
            out_r.copy_(outk[:, : out_r.shape[1]])

    if kv_dtype == "fp8_e4m3" and q.shape[0] > 16:
        # The e4m3 XQA kernel accepts at most 16 query rows; scheduler
        # iterations can mix a large prefill with decode rows.
        for row_start in range(0, q.shape[0], 16):
            row_end = min(row_start + 16, q.shape[0])
            _run(
                q[row_start:row_end],
                logical_indices[row_start:row_end],
                token_to_req[row_start:row_end],
                query_positions[row_start:row_end],
                out[row_start:row_end],
            )
    else:
        _run(q, logical_indices, token_to_req, query_positions, out)
    out.mul_(torch.sigmoid(output_gate_view))


_page4_audit_done = False


def _qsa_page4_audit(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    out_pages: torch.Tensor,
    out_masks: torch.Tensor,
    out_seq: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """One-shot in-engine check of the page4 output against a naive fp32
    reference and the Triton route on the first 8 rows (group 0).

    Localizes engine-level page4 failures: a big |p4 - ref| with a small
    |t8 - ref| means the page4 mapping (strides/positions/plan) is wrong;
    a small |p4 - ref| means the divergence comes from the 2050-vs-2051
    column ABI difference or fp16 rounding.
    """
    rows = min(8, q.shape[0])
    d = q.shape[2]
    logger.info(
        "AUDIT k=%s/%s v=%s/%s",
        tuple(k_cache.shape), tuple(k_cache.stride()),
        tuple(v_cache.shape), tuple(v_cache.stride()),
    )
    logger.info(
        "AUDIT bt=%s/%s t2r[0:%d]=%s count[0:%d]=%s",
        tuple(block_table.shape), tuple(block_table.stride()),
        rows, token_to_req[:rows].tolist(), rows,
        logical_indices[:rows, -1].tolist(),
    )
    logger.info(
        "AUDIT qp[0:%d]=%s seq_lens[:4]=%s",
        rows, query_positions[:rows].tolist(), sequence_lengths[:4].tolist(),
    )
    # The planner pads empty category slots with physical microblock 0
    # (= block 0, tokens 0-3). If that region is non-finite in this engine's
    # pool, the fwd's 0-weight x NaN-V product poisons the output. Dump it.
    b0_k = k_cache[0, 0:4, 0, :]
    b0_v = v_cache[0, 0:4, 0, :]
    logger.info(
        "AUDIT padtarget blk0: micro0 k_finite=%s v_finite=%s whole-blk0 "
        "k_finite=%s v_finite=%s k_micro0[:6]=%s",
        bool(torch.isfinite(b0_k).all().item()),
        bool(torch.isfinite(b0_v).all().item()),
        bool(torch.isfinite(k_cache[0, :, 0, :]).all().item()),
        bool(torch.isfinite(v_cache[0, :, 0, :]).all().item()),
        [round(x, 4) for x in k_cache[0, 0, 0, :6].tolist()],
    )
    page_size = k_cache.shape[1]
    ref = torch.empty_like(out[:rows])
    for r in range(rows):
        count = int(logical_indices[r, -1].item())
        toks = logical_indices[r, :count]
        pages = (toks // page_size).to(torch.int32)
        offs = (toks % page_size).to(torch.int32)
        phys = block_table[int(token_to_req[r].item()), pages]
        kt = k_cache[phys, offs, 0, :].float()
        vt = v_cache[phys, offs, 0, :].float()
        s = q[r].float() @ kt.T * (d ** -0.5)
        ref[r] = s.softmax(-1) @ vt
    logger.info(
        "AUDIT |p4 - ref| = %.4f",
        (out[:rows].float() - ref).abs().max().item(),
    )

    out8 = torch.empty_like(q[:rows])
    _qsa_triton_body(
        q[:rows], k_cache, v_cache, logical_indices[:rows], block_table,
        token_to_req[:rows], True, out8, output_gate_view[:rows],
        k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), d ** -0.5, 1.0,
    )
    gate_eff = torch.sigmoid(output_gate_view[:rows].float())  # per (row, head, dim)
    ref_gated = (ref * gate_eff).to(out.dtype)
    p4_gated = (out[:rows].float() * gate_eff).to(out.dtype)
    logger.info(
        "AUDIT |t8 - ref_gated| = %.4f",
        (out8.float() - ref_gated.float()).abs().max().item(),
    )
    logger.info(
        "AUDIT |p4_gated - t8| = %.4f",
        (p4_gated.float() - out8.float()).abs().max().item(),
    )
    # Pinpoint dump: row 0 (count=1, single token) must be exact if all
    # three routes read the same KV row.
    phys0 = int(block_table[0, 0].item())
    v0 = v_cache[phys0, 0, 0, :6].tolist()
    nf = torch.nonzero(~torch.isfinite(out[:rows].float()))
    nf_coords = (
        [(int(a), int(b), int(c)) for a, b, c in nf[:12].tolist()]
        if nf.numel()
        else []
    )
    # Per-(row, head) histogram: is the NaN count uniform across the 48
    # row-head pairs, and which dims carry it?
    nf_mask = ~torch.isfinite(out[:rows].float())  # [rows, heads, dim]
    per_rh = nf_mask.sum(-1).tolist()
    dims_row0 = [int(x) for x in torch.nonzero(
        nf_mask[0, 0]).flatten().tolist()][:24]
    logger.info(
        "AUDIT p4 nonfinite: count=%d coords12=%s",
        int(nf_mask.sum().item()), nf_coords,
    )
    logger.info("AUDIT nonfinite per(row,head)=%s", per_rh)
    logger.info("AUDIT row0head0 nan dims=%s", dims_row0)
    # What did the planner actually emit for group 0? lse is [cap*8, heads]
    # (2D): row = query index, col = head index.
    logger.info(
        "AUDIT plan g0: seq=%d pages[:12]=%s masks[:12]=%s lse[q0,:4]=%s "
        "q0_finite=%s out_finite_row0=%s",
        int(out_seq[0].item()),
        out_pages[0, :12].tolist(),
        [hex(x) for x in out_masks[0, :12].tolist()],
        [round(x, 4) for x in lse[0, :4].tolist()],
        bool(torch.isfinite(q[0]).all().item()),
        bool(torch.isfinite(out[0]).all().item()),
    )
    logger.info(
        "AUDIT row0: p4=%s t8=%s ref=%s v_cache[bt0,0,:6]=%s bt[:6]=%s "
        "idx0[:4]=%s count=%d p4_finite=%s",
        [round(x, 5) for x in out[0, 0, :6].tolist()],
        [round(x, 5) for x in out8[0, 0, :6].tolist()],
        [round(x, 5) for x in ref[0, 0, :6].tolist()],
        [round(x, 5) for x in v0],
        block_table[0, :6].tolist(),
        logical_indices[0, :4].tolist(),
        int(logical_indices[0, -1].item()),
        bool(torch.isfinite(out[0]).all().item()),
    )
    # Multi-group sweep: the 2048-token audits only ever exercised group 0
    # (vis 1..8). Large-window groups (wide vis, many pages, blocks beyond
    # the first) were never checked — L3's 8192 prefill degeneration starts
    # at the first generated token, which depends on the LAST group's rows.
    ngroups = q.shape[0] // 8
    for g in sorted({0, ngroups // 2, ngroups - 1}):
        g0, g1 = 8 * g, 8 * (g + 1)
        ref_g = torch.empty_like(out[g0:g1])
        for r in range(g0, g1):
            count = int(logical_indices[r, -1].item())
            toks = logical_indices[r, :count]
            pages = (toks // page_size).to(torch.int32)
            offs = (toks % page_size).to(torch.int32)
            phys = block_table[int(token_to_req[r].item()), pages]
            kt = k_cache[phys, offs, 0, :].float()
            vt = v_cache[phys, offs, 0, :].float()
            s = q[r].float() @ kt.T * (d ** -0.5)
            ref_g[r - g0] = s.softmax(-1) @ vt
        nf_g = int(
            torch.nonzero(~torch.isfinite(out[g0:g1].float())).numel()
        )
        logger.info(
            "AUDIT g%d: |p4-ref|=%.4f nonfinite=%d counts=%s seq=%d "
            "pages[:4]=%s masks[:4]=%s",
            g,
            (out[g0:g1].float() - ref_g).abs().max().item(),
            nf_g,
            logical_indices[g0:g1, -1].tolist(),
            int(out_seq[g].item()),
            out_pages[g, :4].tolist(),
            [hex(x) for x in out_masks[g, :4].tolist()],
        )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    *,
    output_gate: torch.Tensor,
    query_positions: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    An fp8 cache is a native e4m3 view on sm_89+ and the raw uint8 storage
    below sm_89 (no Triton fp8e4nv pointer type there); the kernel dequantizes
    either on load. k_scale/v_scale are the layer's per-tensor dequant scales
    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
    pre-multiplied into the softmax scale and v_scale becomes the kernel's
    output scale, so the kernel needs no device scale buffers.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the top of the config table; see
    _select_config.
    """
    global _qsa_entry_probe_count
    if _qsa_entry_probe_count < 4 and q.dim() == 3 and int(q.shape[0]) >= 512:
        _qsa_entry_probe_count += 1
        logger.info(
            "QSA entry probe: rows=%d prefill_cfg=%s q=%s k_cache=%s/%s "
            "strides=%s block=%s token_to_req=%s logical=%s",
            int(q.shape[0]), use_prefill_config, tuple(q.shape),
            tuple(k_cache.shape), k_cache.dtype, tuple(k_cache.stride()),
            tuple(block_table.shape),
            tuple(token_to_req.shape), tuple(logical_indices.shape))
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    is_fp8 = k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    # nvfp4 KV arrives as the uint8 cell pool ([P, 2, N, 144], per-token cell
    # [K data 128 | K scale 16 | V data 128 | V scale 16]); the page4-TP8
    # readers decode it in-kernel, so the packed last dim is not the head dim.
    is_nvfp4 = k_cache.dtype == torch.uint8
    if q.shape[1] % k_cache.shape[2] or (
        not is_nvfp4 and q.shape[2] != k_cache.shape[3]
    ):
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert k_cache.dtype == v_cache.dtype
    if is_fp8 or is_nvfp4:
        # Same query dtypes as the unquantized branch: this deployment's
        # QSA queries are FP16, and the page4-TP8 reader requires FP16 Q.
        assert q.dtype in (torch.bfloat16, torch.float16)
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
        if k_cache.dtype == torch.uint8:
            # _maybe_dequant's byte decode yields x * 2^-8; fold 2^8 into the
            # scales so the decode needs no multiply.
            softmax_scale *= 256.0
            output_scale *= 256.0
        else:
            assert k_cache.dtype == torch.float8_e4m3fn
    else:
        assert q.dtype in (torch.bfloat16, torch.float16)
        assert k_cache.dtype in (torch.bfloat16, torch.float16)
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    # C-2 M2: TP8 page4 route (env-gated, default off). Checked before the
    # FMA arm so the in-service A/B can swap FMA -> page4-TP8 without
    # touching SM70_QSA_FMA.
    _page4_tp8_ok = _qsa_page4_tp8_eligible(
        q,
        k_cache,
        logical_indices,
        block_table,
        query_positions,
        sequence_lengths,
        use_prefill_config,
    )
    global _qsa_gate_probe_count
    if (
        not _page4_tp8_ok
        and _qsa_gate_probe_count < 8
        and use_prefill_config
        and q.shape[0] >= 512
    ):
        # Reject-only diagnostic (first 8): a TP8-gate rejection is what pins
        # prefill to the fallback kernel, and the TP4 gate cannot accept this
        # model (heads==6, but QSA heads are never TP-sharded).
        _qsa_gate_probe_count += 1
        logger.info(
            "QSA page4 gate reject: heads=%d logical=%d k_cache=%s "
            "strides=%s block=%s rows=%d rows%%8=%d prefill=%s pos=%s "
            "seqlen=%s",
            q.shape[1], logical_indices.shape[1], tuple(k_cache.shape),
            tuple(k_cache.stride()), tuple(block_table.shape),
            int(q.shape[0]), int(q.shape[0]) % _QSA_PAGE4_QUERIES,
            use_prefill_config, query_positions is not None,
            sequence_lengths is not None,
        )
    if _page4_tp8_ok:
        _qsa_page4_tp8_route(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            out,
            output_gate_view,
            k_scale or 1.0,
            v_scale or 1.0,
        )
        return out

    # The 6-head (TP4) prefill route must be decided before the nvfp4/fp8
    # fallback below: that fallback deliberately does not look at
    # use_prefill_config, so with the order inherited from the tree
    # snapshot every TP4 prefill was served by the decode XQA batch route
    # (313632 CTAs, ~71 ms per 96-row call) and this gate was unreachable.
    # It only accepts single-request prefill (block_table rows == 1), so
    # decode and multi-request batches keep the fallback path unchanged.
    if _qsa_page4_eligible(
        q,
        k_cache,
        logical_indices,
        block_table,
        query_positions,
        sequence_lengths,
        use_prefill_config,
    ):
        logger.info_once("QSA arm: TP4 6-head page4 (grouped verify)")
        _qsa_page4_route(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            out,
            output_gate_view,
            k_scale=k_scale or 1.0,
            v_scale=v_scale or 1.0,
        )
        return out

    # FP8-KV rows the TP8 route did not take (decode, unaligned prefill)
    # FP8/nvfp4-KV rows the TP8 route did not take (decode, unaligned
    # prefill) run through the Flash-V100 XQA batch route. The table kernel
    # is compiled eagerly here on the first call so decode capture never
    # JITs inside the graph.
    if is_fp8 or is_nvfp4:
        _qsa_xqa_ensure_table_compiled(
            q,
            k_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
        )
        if _qsa_xqa_page4_eligible(
            q,
            k_cache,
            logical_indices,
            block_table,
            query_positions,
            sequence_lengths,
        ):
            logger.info_once(
                "QSA arm: nvfp4/fp8 XQA batch fallback (decode or "
                "unaligned/multi-request prefill)"
            )
            _qsa_xqa_page4_route(
                q,
                k_cache,
                v_cache,
                logical_indices,
                block_table,
                token_to_req,
                query_positions,
                sequence_lengths,
                out,
                output_gate_view,
                k_scale or 1.0,
                v_scale or 1.0,
            )
            return out

    # P1 FMA prefill kernel: +10.8% prefill vs the Triton path. The early
    # live-only 0/NaN bug was root-caused to strided, non-contiguous KV views
    # (flat addressing read wrong addresses); fixed via .contiguous() in
    # qsa_prefill_sm70.prefill_fma_forward. Gated by SM70_QSA_FMA=1 as an ops
    # valve (off by default to stay conservative for upstream sync).
    if (os.environ.get("SM70_QSA_FMA", "0") == "1"
            and use_prefill_config and not is_fp8 and not is_nvfp4
            and q.dtype == torch.float16 and head_dim == 256
            and q.shape[1] <= 4 and out.is_contiguous()
            and k_cache.shape[1] == 784 and k_cache.shape[2] == 1):
        logger.info_once("QSA prefill: SM70 FMA kernel active (SM70_QSA_FMA=1)")
        from vllm.models.qwen4_exp.nvidia.ops import qsa_prefill_sm70 as _qpsm
        _qpsm.prefill_fma_forward(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req,
            out, output_gate_view, logical_indices.shape[1] - 1,
            float(softmax_scale))
        return out


    if is_nvfp4:
        # The nvfp4 pool has no Triton reader; the in-kernel-dequant page4-TP8
        # route is the only way to consume it. Fail loud instead of feeding
        # raw pool bytes to the split-K kernel.
        raise NotImplementedError(
            "QSA nvfp4 KV needs the page4-TP8 route "
            "(SM70_PAGE4_TP8=1 or /tmp/page4_tp8_mode)"
        )

    if (
        k_cache.dtype == torch.float8_e4m3fn
        and _is_sm70()
    ):
        # sm_70 Triton cannot compile fp8e4nv; the FMA arm above already
        # skips fp8, so on this architecture the e4m3 KV has no reader but
        # the page4-TP8 route. Fail loud at the call site instead of dying
        # deep in Triton with a CompilationError.
        raise NotImplementedError(
            "QSA fp8-e4m3 KV on sm_70 needs the page4-TP8 route "
            "(SM70_PAGE4_TP8=1 or /tmp/page4_tp8_mode); the Triton arm "
            "cannot compile fp8e4nv on this architecture"
        )

    return _qsa_triton_body(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config,
        out,
        output_gate_view,
        is_fp8,
        softmax_scale,
        output_scale,
    )


def _qsa_triton_body(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    is_fp8: bool,
    softmax_scale: float,
    output_scale: float,
) -> torch.Tensor:
    """The Triton split-K GQA route over paged K/V caches (default path)."""
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    block_n, partial_warps, num_tiles, num_splits = _select_config(
        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width, is_fp8
    )

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        softmax_scale,
        output_scale,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=_QSA_NUM_STAGES,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def warmup_qsa_sparse_paged_attention(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile every production-reachable split-K/merge specialization."""
    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    # An fp8 cache is allocated as uint8 storage; warm the dtype the impl hands
    # over: an e4m3 view on sm_89+, the raw bytes below.
    is_fp8 = kv_cache.dtype == torch.uint8
    cache_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype
    if is_fp8 and _is_sm70():
        # sm_70 Triton cannot compile fp8e4nv; on sm_70 the in-service fp8
        # reader is the page4-TP8 route, so the fp8 Triton-arm warmup is
        # both uncompilable (e4m3) and speculative (e5m2's runtime variant
        # JITs on first use instead).
        logger.info_once(
            "QSA fp8 KV on sm_70: skipping Triton-arm warmup "
            "(fp8e4nv is not compilable on this architecture)")
        return ()
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)

    # Every config the dispatch can pick for this group size.
    profiles = {
        _select_config(
            num_rows, num_kv_heads, use_prefill_config, selection_width, is_fp8
        )
        for num_rows in range(1, 8193)
        for use_prefill_config in (False, True)
    }

    # Scalars constant per deployment get their real values (their divisibility
    # specialization is wanted); the batch-varying ones are do_not_specialize'd
    # on the kernels, so any value here compiles the only variant.
    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        key_cache.dtype, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(key_cache.shape),
        strides=tuple(key_cache.stride()),
    )
    v_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    # +1: the packed buffer's trailing count column.
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        key_cache.dtype, shape=(num_rows, num_query_heads, head_dim)
    )
    # The output gate is mandatory at runtime; warm the gated specialization.
    output_gate_ptr = TritonWarmupTensor(
        key_cache.dtype, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])

    warmed = []
    for block_n, warps, num_tiles, num_splits in sorted(profiles):
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )
        _qsa_sparse_paged_gqa_splitk_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            indices_ptr,
            block_table_ptr,
            token_to_req_ptr,
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            1.0,
            1.0,
            output_gate_ptr,
            row_stride,
            head_stride,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            selection_width + 1,
            block_table.stride(0),
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            num_cache_blocks,
            num_requests,
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=warps,
            num_stages=2,
            grid=(num_rows, num_kv_heads, num_splits),
        )
        if num_splits > 1:
            _qsa_merge_splitk_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                output_gate_ptr,
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                num_warps=2,
                num_stages=1,
                grid=(num_rows, num_query_heads),
            )
        warmed.append((block_n, num_splits, warps))
    return tuple(warmed)


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""
    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""
    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "qsa_compress_groups_with_ratio",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "warmup_qsa_sparse_paged_attention",
]
