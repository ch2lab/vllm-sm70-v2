# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA index selection."""

import math
import os

import torch

import vllm.envs as envs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


# Triton on SM70 cannot name the fp8e4nv type at all (only e5m2 is accepted),
# so a quantised indexer cache is passed as raw bytes and dequantised with a
# table lookup before any arithmetic. That keeps e4m3's precision instead of
# falling back to e5m2's two mantissa bits.
_FP8_LUT: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_NVFP4_PAIR_LUT: dict[torch.device, torch.Tensor] = {}
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _nvfp4_pair_lut(device: torch.device) -> torch.Tensor:
    """[code << 8 | e4m3 scale byte] -> value * scale as fp16 (4096 entries).

    One L1-resident table (8 KiB) replaces the per-element exponent maths:
    E2M1 has 16 levels and the block scale is an e4m3 byte, so every product
    the dequant can produce is already in here.
    """
    table = _NVFP4_PAIR_LUT.get(device)
    if table is None:
        levels = torch.tensor(_E2M1_LEVELS, device=device, dtype=torch.float32)
        scales = (
            torch.arange(256, dtype=torch.uint8, device=device)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        table = (levels[:, None] * scales[None, :]).reshape(-1).to(torch.float16)
        _NVFP4_PAIR_LUT[device] = table
    return table


def _is_nvfp4_index_cache(k_cache: torch.Tensor, head_dim: int) -> bool:
    return (
        k_cache.dtype == torch.uint8
        and k_cache.shape[-1] == head_dim // 2 + head_dim // 16
    )


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _fp8_needs_table(cache: torch.Tensor) -> bool:
    """True when an e4m3 cache must travel as bytes (Volta has no fp8e4nv)."""
    return cache.dtype == torch.float8_e4m3fn and (
        torch.cuda.get_device_capability(cache.device.index)[0] < 8
    )


def _fp8_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Byte -> fp16 table for whichever fp8 format the cache holds."""
    key = (device, dtype)
    table = _FP8_LUT.get(key)
    if table is None:
        table = (
            torch.arange(256, dtype=torch.uint8, device=device)
            .view(dtype)
            .to(torch.float16)
        )
        _FP8_LUT[key] = table
    return table


@triton.jit
def _fp8_byte_value(raw, lut_ptr):
    """e4m3 byte -> fp16 in registers (no table gather).

    Volta's Triton rejects fp8e4nv outright, so an e4m3 index cache is carried
    as bytes and rebuilt here; e5m2 takes the native cast instead. e4m3 and
    fp16 differ only by a constant exponent bias (7 vs 15) and a 4-bit shift
    of the mantissa, so the normal case is two shifts and an or. Subnormals
    (e == 0) are m * 2^-9 and exact in fp16, so they take one multiply.
    """
    b = raw.to(tl.int32)
    e = (b >> 3) & 0xF
    m = b & 0x7
    sign = (b & 0x80) << 8
    sub = (m.to(tl.float32) * 0.001953125).to(tl.float16)
    sub_bits = sub.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where(e == 0, sub_bits, ((e + 8) << 10) | (m << 7)) | sign
    return bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _dequant_nvfp4(raw, sbyte, pair_ptr, half):
    """NVFP4 nibble -> fp16 through the (code, e4m3 scale) product table."""
    code = (raw.to(tl.int32) >> (4 * (half % 2))) & 0xF
    return tl.load(pair_ptr + ((code << 8) | sbyte.to(tl.int32)))


@triton.jit
def _load_nvfp4_rows(
    row0, pair_ptr, row_ok, BLOCK_R: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """Read BLOCK_R packed rows as vectors and expand them in registers.

    One row is 8 block scales + 64 nibble-packed bytes.  Loading the row
    element-by-element issues three gathers per value (byte, broadcast scale,
    table); loading the two vectors and spreading them over the 128 lanes
    issues 72 loads per row instead, which is what the row count scales on.
    """
    n8: tl.constexpr = HEAD_DIM // 16
    n64: tl.constexpr = HEAD_DIM // 2
    d64 = tl.arange(0, n64)
    d8 = tl.arange(0, n8)
    raw = tl.load(
        row0[:, None] + n8 + d64[None, :],
        mask=row_ok[:, None],
        other=0,
        eviction_policy="evict_first",
    )
    sb = tl.load(
        row0[:, None] + d8[None, :],
        mask=row_ok[:, None],
        other=0,
        eviction_policy="evict_first",
    )
    # byte j carries the scale of dims [16*(j//8), 16*(j//8)+16)
    sb64 = tl.reshape(
        tl.broadcast_to(
            tl.reshape(sb, (BLOCK_R, n8, 1)),
            (BLOCK_R, n8, HEAD_DIM // (2 * n8)),
        ),
        (BLOCK_R, n64),
    )
    idx = sb64.to(tl.int32)
    lo = tl.load(pair_ptr + (((raw.to(tl.int32) & 0xF) << 8) | idx))
    hi = tl.load(pair_ptr + ((((raw.to(tl.int32) >> 4) & 0xF) << 8) | idx))
    return tl.reshape(tl.join(lo, hi), (BLOCK_R, HEAD_DIM))


_TOPK_WORKSPACE_BYTES = 1024 * 1024
_DECODE_BLOCK_N = 64

# Per-dtype launch shapes for the Volta decode scan, measured on 16.4k columns
# with two query rows (min of 30, logits bit-identical to the default config):
# bf16 59.5 us, fp8 52.8, e4m3 66.8, nvfp4 70.9 -- versus 78/109/73 us for the
# GB300-tuned default. The kernel is latency-bound, so it wants either many
# one-warp programs or fewer wide ones depending on how much work each element
# needs to decode.
_SM70_DECODE_SCAN = {
    "bf16": (64, 2, 2, 4),
    "fp8": (32, 2, 1, 1),
    "e4m3": (16, 2, 3, 1),
    "nvfp4": (64, 2, 2, 2),
}


def _sm70_decode_scan_config(
    cache: torch.Tensor, block_n: int, tpp: int, stages: int, warps: int
) -> tuple[int, int, int, int]:
    """Launch shape for the decode scan, or the defaults off Volta."""
    if not current_platform.is_device_capability(70):
        return block_n, tpp, stages, warps
    if cache.dtype == torch.uint8:
        key = "nvfp4"
    elif cache.dtype == torch.float8_e4m3fn:
        key = "e4m3"
    elif cache.dtype == torch.float8_e5m2:
        key = "fp8"
    else:
        key = "bf16"
    return _SM70_DECODE_SCAN[key]

# SM70 cuBLAS indexer path (ported from 1Cat-vLLM's production SM70 stack,
# 2026-10-02). For the exact single-request fp16 shape, gather this
# request's paged compressed keys once, score with cuBLAS (Volta HMMA) in
# column chunks, fuse ReLU + head-sum + visibility in a Triton epilogue,
# then top-k. Replaces the paged-Triton scan (51.6% of 65K prefill kernel
# time) without a device-side early-exit kernel. Fallbacks keep the scan.
_SM70_INDEXER_CUBLAS = os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS", "1") == "1"
_SM70_INDEXER_SCORE_TILE_BYTES = (
    int(os.getenv("VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB", "64")) * 1024 * 1024
)
_SM70_INDEXER_CUBLAS_MIN_ROWS = int(
    os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS", "512")
)
_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS = int(
    os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS", str(1024**2))
)
# Exact tile-max prefilter for the selection stage (see
# csrc/qsa_prefilter_topk.cuh). The tile maxes are produced for free by the
# score epilogue; the selector then only scans the 512 tiles that can contain
# a top-k element instead of the whole row.
_SM70_INDEXER_PREFILTER = os.getenv("VLLM_SM70_QSA_PREFILTER", "1") == "1"
_SM70_INDEXER_PREFILTER_TILE = 16
_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024


@triton.jit
def _qsa_gather_single_request_keys_kernel(
    cache_ptr,
    page_table_ptr,
    keys_ptr,
    valid_ptr,
    lut_ptr,
    stride_cache_page,
    stride_cache_token,
    stride_cache_dim,
    stride_table_page,
    stride_keys_row,
    columns,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DEQUANT: tl.constexpr,
    NVFP4: tl.constexpr,
    FP8_LUT: tl.constexpr,
) -> None:
    column = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    logical_page = column // PAGE_SIZE
    page_offset = column % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr + logical_page * stride_table_page,
        mask=(column < columns) & (logical_page < PAGE_TABLE_WIDTH),
        other=-1,
    )
    valid = (column < columns) & (physical_page >= 0) & (physical_page < num_pages)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    row_base = (
        cache_ptr + safe_page * stride_cache_page + page_offset * stride_cache_token
    )
    if NVFP4:
        in_row = valid & (dims < HEAD_DIM)
        raw = tl.load(row_base + (HEAD_DIM // 16) + dims // 2, mask=in_row, other=0)
        sbyte = tl.load(row_base + dims // 16, mask=in_row, other=0)
        key = _dequant_nvfp4(raw, sbyte, lut_ptr, dims)
    elif FP8_LUT:
        # uint8 view of an e4m3 cache
        key = _fp8_byte_value(
            tl.load(
                row_base + dims * stride_cache_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0,
            ),
            lut_ptr,
        )
    else:
        key = tl.load(
            row_base + dims * stride_cache_dim,
            mask=valid & (dims < HEAD_DIM),
            other=0.0,
        )
        if DEQUANT:
            key = key.to(tl.float16)
    tl.store(
        keys_ptr + column * stride_keys_row + dims,
        key,
        mask=(column < columns) & (dims < HEAD_DIM),
    )
    tl.store(valid_ptr + column, valid, mask=column < columns)


@triton.jit
def _qsa_relu_headsum_visible_kernel(
    score_ptr,
    visible_ptr,
    key_valid_ptr,
    logits_ptr,
    tile_max_ptr,
    stride_score_row,
    stride_score_column,
    stride_logits_row,
    stride_tile_row,
    width,
    column_start,
    score_divisor,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    absolute_columns = column_start + columns
    visible = tl.load(visible_ptr + row)
    key_valid = tl.load(
        key_valid_ptr + absolute_columns,
        mask=columns < width,
        other=0,
    ).to(tl.int1)
    live = (columns < width) & (absolute_columns < visible) & key_valid
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in range(NUM_HEADS):
        values = tl.load(
            score_ptr
            + (row * NUM_HEADS + head) * stride_score_row
            + columns * stride_score_column,
            mask=columns < width,
            other=0.0,
        ).to(tl.float32)
        score += tl.maximum(values, 0.0)
    logit = tl.where(live, score / score_divisor, -float("inf"))
    tl.store(
        logits_ptr + row * stride_logits_row + absolute_columns,
        logit,
        mask=columns < width,
    )
    # Per-TILE maximum of the logits this program just produced. The tile-max
    # prefilter selects its candidates from these, so they must be the same
    # values the selector would have read back (masked lanes are -inf).
    tile_max = tl.max(tl.reshape(logit, (BLOCK_N // TILE, TILE)), axis=1)
    tile_ids = (
        column_start + tl.program_id(1) * BLOCK_N
    ) // TILE + tl.arange(0, BLOCK_N // TILE)
    tl.store(
        tile_max_ptr + row * stride_tile_row + tile_ids,
        tile_max,
        mask=tile_ids < stride_tile_row,
    )


def _qsa_indexer_cublas_shape_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> bool:
    """Whether the exact Qwen3.8 single-request index shape can use cuBLAS.

    FP16 runs directly; BF16 is accepted because its values are RMSNormed
    (8-bit mantissa fits the 10-bit FP16 mantissa exactly, no overflow risk
    at RMS=1) and V100 has no BF16 HMMA.
    """

    fp8_cache = _is_fp8(k_cache.dtype)
    # A packed nvfp4 cache is uint8 with a 72-byte row, so both the dtype and
    # the last-dim check have to accept it or prefill silently falls back from
    # cuBLAS to the Triton scan (measured 8x slower: 1102 vs 8858 tok/s).
    packed = _is_nvfp4_index_cache(k_cache, q.shape[2])
    return (
        q.dtype in (torch.float16, torch.bfloat16)
        and (
            k_cache.dtype == q.dtype
            or ((fp8_cache or packed) and q.dtype == torch.float16)
        )
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and k_cache.ndim == 4
        and (
            (k_cache.shape[2] == 1 and k_cache.shape[3] == 72)
            if packed
            else k_cache.shape[2:] == (1, 128)
        )
        and page_table.ndim == 2
        and page_table.shape[0] == 1
    )


def _qsa_gather_single_request_keys(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    columns: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = _is_fp8(k_cache.dtype)
    nvfp4 = _is_nvfp4_index_cache(k_cache, head_dim)
    fp8_lut = _fp8_needs_table(k_cache)
    cache = k_cache.view(torch.uint8) if fp8_lut else k_cache
    keys = torch.empty(
        (columns, head_dim),
        dtype=torch.float16 if (fp8 or nvfp4) else k_cache.dtype,
        device=k_cache.device,
    )
    valid = torch.empty(columns, dtype=torch.uint8, device=k_cache.device)
    if columns:
        _qsa_gather_single_request_keys_kernel[(columns,)](
            cache,
            page_table[0],
            keys,
            valid,
            (
                _nvfp4_pair_lut(k_cache.device)
                if nvfp4
                else _fp8_lut(k_cache.device, k_cache.dtype)
                if fp8
                else keys
            ),
            cache.stride(0),
            cache.stride(1),
            cache.stride(3),
            page_table.stride(1),
            keys.stride(0),
            columns,
            k_cache.shape[0],
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            # HEAD_DIM is the logical key width, not the row size: a packed
            # nvfp4 row is 72 bytes for 128 dims.
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            DEQUANT=fp8,
            NVFP4=nvfp4,
            FP8_LUT=fp8_lut,
            num_warps=4,
        )
    return keys, valid


def _qsa_mqa_cublas(
    q: torch.Tensor,
    keys: torch.Tensor,
    key_valid: torch.Tensor,
    visible: torch.Tensor,
    tile_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one request with Volta Tensor Cores and a fused QSA epilogue."""

    rows, num_heads, head_dim = q.shape
    columns = keys.shape[0]
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    if not rows or not columns:
        return logits, tile_max

    q2 = q.reshape(rows * num_heads, head_dim)
    bytes_per_column = max(1, q2.shape[0] * 4)
    tile_columns = max(
        256,
        min(columns, _SM70_INDEXER_SCORE_TILE_BYTES // bytes_per_column),
    )
    tile_columns = max(256, tile_columns // 256 * 256)
    for column_start in range(0, columns, tile_columns):
        column_stop = min(column_start + tile_columns, columns)
        width = column_stop - column_start
        # FP16 inputs, FP32 output: keeps Volta HMMA and avoids an FP16
        # round trip before top-k.
        scores = torch.mm(
            q2,
            keys[column_start:column_stop].t(),
            out_dtype=torch.float32,
        )
        _qsa_relu_headsum_visible_kernel[
            (rows, triton.cdiv(width, 256))
        ](
            scores,
            visible,
            key_valid,
            logits,
            tile_max,
            scores.stride(0),
            scores.stride(1),
            logits.stride(0),
            tile_max.stride(0),
            width,
            column_start,
            math.sqrt(head_dim),
            NUM_HEADS=num_heads,
            BLOCK_N=256,
            TILE=_SM70_INDEXER_PREFILTER_TILE,
            num_warps=4,
        )
    return logits, tile_max


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    lut_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    DEQUANT: tl.constexpr,
    NVFP4: tl.constexpr,
    FP8_LUT: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        # Pool bounds: an entry outside the allocated pool (stale block id
        # from a recycled request slot, or a growth page the metadata counts
        # but the table has not mapped) must not be indexed into k_cache --
        # it faults the process (EXP-12f 2026-10-16). Index page 0 with
        # zeros instead so the column scores degrade to 0.
        page_ok = (
            live
            & (physical_page >= 0)
            & (physical_page < num_cache_blocks)
        )
        safe_page = tl.where(page_ok, physical_page, 0)
        in_row = page_ok[:, None] & (dims[None, :] < HEAD_DIM)
        row_base = (
            k_cache_ptr
            + safe_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
        )
        if NVFP4:
            row0 = (
                k_cache_ptr
                + safe_page.to(tl.int64) * stride_cache_block
                + page_offset * stride_cache_token
            )
            keys = _load_nvfp4_rows(row0, lut_ptr, page_ok, BLOCK_N, HEAD_DIM)
        elif FP8_LUT:
            # uint8 view of an e4m3 cache
            keys = _fp8_byte_value(
                tl.load(
                    row_base + dims[None, :],
                    mask=in_row,
                    other=0,
                    eviction_policy="evict_first",
                ),
                lut_ptr,
            )
        else:
            keys = tl.load(
                row_base + dims[None, :],
                mask=in_row,
                other=0.0,
                eviction_policy="evict_first",
            )
            if DEQUANT:
                keys = keys.to(tl.float16)
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    lut_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    DEQUANT: tl.constexpr,
    NVFP4: tl.constexpr,
    FP8_LUT: tl.constexpr,
) -> None:
    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        # Pool bounds: an entry outside the allocated pool (stale block id
        # from a recycled request slot, or a growth page the metadata counts
        # but the table has not mapped) must not be indexed into k_cache --
        # it faults the process (EXP-12f 2026-10-16). Index page 0 with
        # zeros instead so the column scores degrade to 0.
        page_ok = (
            live
            & (physical_page >= 0)
            & (physical_page < num_cache_blocks)
        )
        safe_page = tl.where(page_ok, physical_page, 0)
        in_row = page_ok[:, None] & (dims[None, :] < HEAD_DIM)
        row_base = (
            k_cache_ptr
            + safe_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
        )
        if NVFP4:
            row0 = (
                k_cache_ptr
                + safe_page.to(tl.int64) * stride_cache_block
                + page_offset * stride_cache_token
            )
            keys = _load_nvfp4_rows(row0, lut_ptr, page_ok, BLOCK_N, HEAD_DIM)
        elif FP8_LUT:
            # uint8 view of an e4m3 cache
            keys = _fp8_byte_value(
                tl.load(
                    row_base + dims[None, :],
                    mask=in_row,
                    other=0,
                    eviction_policy="evict_first",
                ),
                lut_ptr,
            )
        else:
            keys = tl.load(
                row_base + dims[None, :],
                mask=in_row,
                other=0.0,
                eviction_policy="evict_first",
            )
            if DEQUANT:
                keys = keys.to(tl.float16)
        # With an fp8 cache, SM90 lowers this dot to reduced-precision wgmma
        # accumulation; SM100 tcgen05 accumulates in fp32. The SM90 error
        # (~1e-4 relative) is far below the e4m3 quantization noise floor.
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (columns < OUTPUT_WIDTH) & (is_expanded | is_tail) & (token >= 0)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=columns < OUTPUT_WIDTH,
    )
    # TRAILING COUNT COLUMN: the packed buffer is OUTPUT_WIDTH+1 wide; the
    # last column holds this row's valid-entry count (expanded blocks plus
    # causal tail). It is NOT a token index — the sparse attention kernel
    # reads it as the row's tile-loop bound.
    if tl.program_id(1) == 0:
        tl.store(
            output_ptr + row * stride_output_row + OUTPUT_WIDTH * stride_output_column,
            expanded_count + tail_count,
        )


def _decode_tiles_per_program(num_requests: int, columns: int) -> int:
    programs = num_requests * triton.cdiv(columns, _DECODE_BLOCK_N)
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    columns: int,
) -> tuple[tuple[int, int], ...]:
    profiles: list[tuple[int, int]] = []
    for dql in range(1, max_dql + 1):
        max_reqs = min(max_num_reqs, max_num_batched_tokens // dql)
        requests_by_grouping: dict[int, int] = {}
        for num_requests in range(1, max_reqs + 1):
            tiles_per_program = _decode_tiles_per_program(num_requests, columns)
            requests_by_grouping.setdefault(tiles_per_program, num_requests)
        profiles.extend(
            (dql, num_requests) for num_requests in requests_by_grouping.values()
        )
    return tuple(profiles)


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""
    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
        columns,
    )
    if not profiles:
        return ()

    fp8 = _is_fp8(k_cache.dtype)
    # A uint8 cache is the packed nvfp4 layout; the decode kernel then reads
    # nibbles, so the warmup must compile that branch, not the fp8 cast one.
    nvfp4 = k_cache.dtype == torch.uint8
    fp8_lut = _fp8_needs_table(k_cache)
    k_cache_ptr = TritonWarmupTensor(
        torch.uint8 if (nvfp4 or fp8_lut) else k_cache.dtype,
        shape=tuple(k_cache.shape),
    )
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)
    # Dequantisation table dummy: fp16, matching the real e4m3 LUT.
    lut_ptr = TritonWarmupTensor(torch.float16)
    # Compile exactly the specialisation the decode launcher will pick, or the
    # first real step pays a JIT stall.
    block_n, _, stages, warps = _sm70_decode_scan_config(
        k_cache, _DECODE_BLOCK_N, 0, 2, 1 if fp8 else 2
    )

    for decode_query_len, num_requests in profiles:
        num_rows = decode_query_len * num_requests
        q_ptr = TritonWarmupTensor(
            torch.float16 if (fp8 or nvfp4) else k_cache.dtype,
            shape=(num_rows, num_heads, head_dim),
        )
        logits_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_rows, columns),
        )
        tiles_per_program = _decode_tiles_per_program(num_requests, columns)
        if current_platform.is_device_capability(70):
            tiles_per_program = _SM70_DECODE_SCAN[
                "nvfp4"
                if nvfp4
                else "e4m3"
                if k_cache.dtype == torch.float8_e4m3fn
                else "fp8"
                if k_cache.dtype == torch.float8_e5m2
                else "bf16"
            ][1]
        _qsa_mqa_paged_uniform_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            lut_ptr,
            num_heads * head_dim,
            head_dim,
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            columns,
            1024,  # num_cache_blocks: compile-only placeholder
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DECODE_QUERY_LEN=decode_query_len,
            BLOCK_N=block_n,
            TILES_PER_PROG=tiles_per_program,
            STAGES=stages,
            DEQUANT=fp8,
            NVFP4=nvfp4,
            FP8_LUT=fp8_lut,
            num_warps=warps,
            grid=(
                num_requests,
                triton.cdiv(columns, block_n * tiles_per_program),
            ),
        )
    return profiles


def _prefill_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits_workspace: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]
    assert logits_workspace.is_contiguous()
    assert logits_workspace.numel() >= num_queries * logits_width

    # slice from pre-allocated workspace
    logits = logits_workspace[: num_queries * logits_width].view(
        num_queries, logits_width
    )

    # tuned on GB300
    fp8 = _is_fp8(k_cache.dtype)
    nvfp4 = _is_nvfp4_index_cache(k_cache, q.shape[2])
    fp8_lut = _fp8_needs_table(k_cache)
    cache = k_cache.view(torch.uint8) if fp8_lut else k_cache
    if fp8:
        TILE_R, STAGES, num_warps = 32, 2, 8
    else:
        TILE_R, STAGES, num_warps = 64, 2, 4
    # This kernel's per-block shared memory scales with TILE_R alone (measured on
    # SM70: 64 -> 160 KB, 32 -> 96 KB, 16 -> 64 KB; K_TILES/STAGES change nothing).
    # Volta caps opt-in smem at 96 KB, so 64 cannot launch there; 32 would sit
    # exactly at the limit with no margin, so drop to 16 on small-smem parts.
    # Each logits[row, col] is one fixed-length dot over HEAD_DIM produced by a
    # single program, so shrinking the row tile only reassigns work: no change to
    # accumulation order or values.
    if (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).shared_memory_per_block_optin
        < 163840
    ):
        TILE_R = 16
    BLOCK_N = 64
    K_TILES = 16
    # Offline/experiment override for the MQA score kernel's row tile and
    # pipeline depth (default = the production heuristic above). Larger TILE_R
    # reuses each KV tile across more query rows; on Volta it is bounded by the
    # 96 KiB opt-in shared-memory cap.
    _env_tile_r = int(os.getenv("VLLM_SM70_QSA_MQA_TILE_R", "0") or 0)
    _env_stages = int(os.getenv("VLLM_SM70_QSA_MQA_STAGES", "0") or 0)
    if _env_tile_r > 0:
        TILE_R = _env_tile_r
    if _env_stages > 0:
        STAGES = _env_stages
    _env_block_n = int(os.getenv("VLLM_SM70_QSA_MQA_BLOCK_N", "0") or 0)
    _env_k_tiles = int(os.getenv("VLLM_SM70_QSA_MQA_K_TILES", "0") or 0)
    if _env_block_n > 0:
        BLOCK_N = _env_block_n
    if _env_k_tiles > 0:
        K_TILES = _env_k_tiles
    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), TILE_R),
        triton.cdiv(logits_width, BLOCK_N * K_TILES),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        _nvfp4_pair_lut(k_cache.device)
        if nvfp4
        else _fp8_lut(k_cache.device, k_cache.dtype)
        if fp8
        else logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        num_queries,
        query_offset,
        page_table.shape[1],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=TILE_R,
        BLOCK_N=BLOCK_N,
        K_TILES=K_TILES,
        STAGES=STAGES,
        DEQUANT=fp8,
        NVFP4=nvfp4,
        FP8_LUT=fp8_lut,
        num_warps=num_warps,
    )
    return logits


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Expand compressed blocks and compact the causal tail of the open group."""
    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (query_positions.numel(), block_topk)
    assert visible_blocks.shape == query_positions.shape
    # +1: the packed buffer's trailing column holds each row's valid-entry
    # count (never a token index); the index region below only writes
    # columns [0, output_width).
    assert out.shape == (block_indices.shape[0], output_width + 1)
    column_block = 256
    grid = (block_indices.shape[0], triton.cdiv(output_width, column_block))
    _expand_qsa_indices_kernel[grid](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        *block_indices.stride(),
        *out.stride(),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )


def _topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    block_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
    use_lexicographic: bool = False,
) -> None:
    # similar dispatch logic as DeepSeek indexer
    block_topk = token_topk // compress_ratio
    if use_lexicographic:
        # Exact, deterministic selector for Volta (score descending, index
        # ascending). persistent_topk is racy on near-tie rows once a row
        # has more live blocks than the top-k width.
        torch.ops._C.qsa_lexicographic_topk(
            logits,
            visible_blocks,
            block_indices,
            block_topk,
        )
        return
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score and select compressed blocks for a request-major decode batch.

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        block_indices: Compressed-index output buffer.

    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    assert (
        q.dtype == k_cache.dtype
        or _is_fp8(k_cache.dtype)
        or _is_nvfp4_index_cache(k_cache, q.shape[2])
    ), "Q and the compressed K cache must match"
    if _is_fp8(q.dtype):
        q = q.to(torch.float16)
    num_requests = q.shape[0] // decode_query_len
    assert page_table.shape[0] == num_requests
    assert visible_blocks.shape == (q.shape[0],)

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    tiles_per_program = _decode_tiles_per_program(num_requests, columns)
    block_n = _DECODE_BLOCK_N
    stages = 2
    warps = 0
    fp8 = _is_fp8(k_cache.dtype)
    nvfp4 = _is_nvfp4_index_cache(k_cache, q.shape[2])
    fp8_lut = _fp8_needs_table(k_cache)
    cache = k_cache.view(torch.uint8) if fp8_lut else k_cache
    block_n, tiles_per_program, stages, warps = _sm70_decode_scan_config(
        k_cache, block_n, tiles_per_program, stages, warps
    )
    grid = (
        num_requests,
        triton.cdiv(columns, block_n * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        cache,
        page_table,
        visible_blocks,
        logits,
        (
            _nvfp4_pair_lut(k_cache.device)
            if nvfp4
            else _fp8_lut(k_cache.device, k_cache.dtype)
            if fp8
            else logits
        ),
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=block_n,
        TILES_PER_PROG=tiles_per_program,
        STAGES=stages,
        DEQUANT=fp8,
        NVFP4=nvfp4,
        FP8_LUT=fp8_lut,
        num_warps=warps or (1 if fp8 else 2),
    )
    _topk(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        torch.empty((_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device),
        use_lexicographic=envs.VLLM_SM70_QSA_DECODE_LEX_TOPK,
    )


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Score and select compressed prefill blocks in bounded chunks.

    Args:
        q: Packed prefill query tensor shaped ``[num_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        query_start_loc: Packed prefill query offsets with a terminal offset.
            Offsets may share the base of a larger backing tensor.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        max_query_len: Maximum number of query tokens in one request.
        block_indices: Compressed-index output buffer.
        max_seq_len: Longest context length in the batch this step.

    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert (
        q.dtype == k_cache.dtype
        or _is_fp8(k_cache.dtype)
        or _is_nvfp4_index_cache(k_cache, q.shape[2])
    ), "Q and the compressed K cache must match"
    if _is_fp8(q.dtype):
        q = q.to(torch.float16)
    rows = q.shape[0]
    # Exact, deterministic selector (1Cat's production choice on Volta):
    # persistent_topk is racy on near-tie rows once a row has more live
    # blocks than the top-k width.
    use_lexicographic = (
        token_topk // compress_ratio == 512
        and current_platform.is_device_capability(70)
    )
    if (
        _SM70_INDEXER_CUBLAS
        and current_platform.is_device_capability(70)
        and rows >= _SM70_INDEXER_CUBLAS_MIN_ROWS
        and _qsa_indexer_cublas_shape_supported(q, k_cache, page_table)
    ):
        block_topk = token_topk // compress_ratio
        # Bound the scoring to this request's visible context. max_seq_len
        # is already on the host, so this avoids 1Cat's one scalar sync per
        # call (all_visible.max().item()).
        score_columns = min(
            max(block_topk, triton.cdiv(max_seq_len, compress_ratio)),
            page_table.shape[1] * k_cache.shape[1],
        )
        if rows * score_columns >= _SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS:
            logger.info_once(
                "QSA indexer prefill cuBLAS path active: rows=%d "
                "score_columns=%d (VLLM_SM70_QSA_INDEXER_CUBLAS).",
                rows,
                score_columns,
            )
            keys, key_valid = _qsa_gather_single_request_keys(
                k_cache, page_table, score_columns, q.shape[2]
            )
            # Volta tensor cores take FP16, not BF16; the cast is exact
            # for these RMSNormed indexer values.
            if keys.dtype != torch.float16:
                keys = keys.to(torch.float16)
            rows_per_chunk = max(
                1, _LOGITS_WORKSPACE_BYTES // max(score_columns * 4, 1)
            )
            topk_workspace = torch.empty(
                (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
            )
            block_topk = token_topk // compress_ratio
            prefilter = _SM70_INDEXER_PREFILTER and block_topk == 512
            tile = _SM70_INDEXER_PREFILTER_TILE
            num_tiles = triton.cdiv(score_columns, tile)
            # The score epilogue always emits per-tile maxima (0.25 B per
            # candidate); the prefilter is what consumes them.
            tile_max = torch.empty(
                (rows_per_chunk, num_tiles),
                dtype=torch.float32,
                device=q.device,
            )
            if prefilter:
                span = block_topk * tile
                logger.info_once(
                    "QSA indexer prefilter active: tile=%d span=%d "
                    "tiles_per_row=%d (VLLM_SM70_QSA_PREFILTER).",
                    tile,
                    span,
                    num_tiles,
                )
                # Selection workspaces, sized for the widest row chunk.
                tiles_ws = torch.empty(
                    (rows_per_chunk, block_topk),
                    dtype=torch.int32,
                    device=q.device,
                )
                cand_ws = torch.empty(
                    (rows_per_chunk, span), dtype=torch.float32, device=q.device
                )
                pos_ws = torch.empty_like(tiles_ws)
                tile_len_ws = torch.empty(
                    (rows_per_chunk,), dtype=torch.int32, device=q.device
                )
                span_len_ws = torch.empty_like(tile_len_ws)
            for query_start in range(0, rows, rows_per_chunk):
                query_end = min(query_start + rows_per_chunk, rows)
                query_slice = slice(query_start, query_end)
                chunk = query_end - query_start
                q_chunk = q[query_slice]
                if q_chunk.dtype != torch.float16:
                    q_chunk = q_chunk.to(torch.float16)
                logits, chunk_tile_max = _qsa_mqa_cublas(
                    q_chunk,
                    keys,
                    key_valid,
                    visible_blocks[query_slice],
                    tile_max[:chunk],
                )
                if prefilter:
                    torch.ops._C.qsa_prefilter_topk(
                        logits,
                        chunk_tile_max,
                        visible_blocks[query_slice],
                        tiles_ws[:chunk],
                        cand_ws[:chunk],
                        pos_ws[:chunk],
                        tile_len_ws[:chunk],
                        span_len_ws[:chunk],
                        block_indices[query_slice],
                        tile,
                        block_topk,
                    )
                    continue
                _topk(
                    logits,
                    visible_blocks[query_slice],
                    token_topk,
                    compress_ratio,
                    block_indices[query_slice],
                    topk_workspace,
                    use_lexicographic=use_lexicographic,
                )
            return
    # No row scores beyond cdiv(max_seq_len, compress_ratio) compressed
    # columns. Round up to 64 to keep the logits row stride
    # cooperative_topk-compatible.
    logits_width = round_up(triton.cdiv(max_seq_len, compress_ratio), 64)
    logits_width = min(max(64, logits_width), page_table.shape[1] * k_cache.shape[1])

    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    # always allocate the worst case to avoid memory fragmentation.
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))

    budget_bytes = max(max_logits_bytes, page_table.shape[1] * k_cache.shape[1] * 4)
    logits_workspace = q.new_empty(budget_bytes // 4, dtype=torch.float32)

    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )

    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        query_slice = slice(query_start, query_end)
        logits = _prefill_logits(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            logits_workspace,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
        _topk(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
            use_lexicographic=use_lexicographic,
        )


__all__ = [
    "expand_qsa_block_indices",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_paged_decode",
]
