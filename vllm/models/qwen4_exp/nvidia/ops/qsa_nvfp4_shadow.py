# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVFP4-KV QSA prefill shadow route (S3b candidate #1, REF_MAP bisect 10-13).

The live QSA layers keep an fp16 KV via --kv-cache-dtype-skip-layers because
the clean tree has no QSA reader for the packed NVFP4 cache (the nvfp4-KV QSA
route lived only in the excluded 1Cat lineage). S3a measured the Triton
splitk kernel as KV-byte-cost sensitive (fp8/fp16 = 0.508x at a 2:1 byte
ratio), so the ~3.56x packed-byte saving is the single thickest lever on the
18% gap to the 1Cat 5464 tok/s anchor.

This module is a DEFAULT-OFF shadow (11670: main path byte-identical when
VLLM_SM70_QSA_NVFP4_SHADOW is unset/0). It runs the same Triton splitk
skeleton as ops/qsa.py::_qsa_sparse_paged_gqa_splitk_kernel (splits=1 form)
over the packed NVFP4 paged cache with the dequant in-kernel:

  * KV cache contract = the LBHNC codec layout the SM70 nvfp4 store kernel
    produces (sm70_nvfp4_dequant.py header). With H_kv == 1 (the QSA layer
    geometry) one side block of N tokens is
        [data: N*DD bytes][scale: N*SD bytes],  DD = D//2, SD = D//16,
    i.e. D=256, N=16 -> 2304 B per side block vs 8192 B for the fp16 cache
    (the 3.56x).
  * Dequant semantics = exact inverse of the store kernel
    (csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu):
        x_hat = E2M1_LUT[q] * float(sf_code) * scale
    The per-side layer scale is FOLDED HOST-SIDE like the fp8 arm: the K
    scale into softmax_scale, the V scale into output_scale, so the kernel
    dequant is scale-free.
  * The dequant pattern (even/odd nibble gather + in-register fp16 material,
    whole-tile at once) mirrors sm70_nvfp4_prefill.py /
    sm70_nvfp4_dequant.py (sm_70-safe form); BLOCK_N = 2 pages (32), the
    2026-10-15 V100 sweep winner.

Idea source (11671, idea-level only): 1Cat lineage nvfp4-KV QSA route
(REF_MAP_1CAT_STRATA.md S2/H1); no code ported from any reference repo.
Offline gates: sm70_tests/check_qsa_nvfp4_shadow.py (CPU structural),
sm70_tests/bench_qsa_nvfp4_shadow.py (GPU parity/invariance/perf, window).
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    _nvfp4_e2m1_decode,
    _nvfp4_e4m3_decode,
)

_ENV = "VLLM_SM70_QSA_NVFP4_SHADOW"


def nvfp4_shadow_enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _packed_block_bytes(page: int, head_dim: int) -> int:
    """One side block (H_kv==1): N*(DD + SD) bytes, DD=D//2, SD=D//16."""
    return page * (head_dim // 2 + head_dim // 16)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_nvfp4_splitk_kernel(
    q_ptr,
    k_cache_ptr,  # u8 [B, N, 1, DD+SD] (LBHNC side block, H_kv==1)
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    output_ptr,
    softmax_scale,  # head_dim**-0.5 * k_scale (host-folded)
    output_scale,  # v_scale (host-folded)
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,  # bytes per side block = N * (DD + SD)
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    DD: tl.constexpr = HEAD_DIM // 2
    SD: tl.constexpr = HEAD_DIM // 16

    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count.
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + head_offsets[:, None] * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # log2 units for the exp2-based online softmax; the K layer scale is
    # already folded into softmax_scale on the host.
    score_scale = softmax_scale * 1.4426950408889634
    # Per-dim nibble/scale-group selectors, broadcast to [BLOCK_N, D] for
    # the tl.gather form the in-tree nvfp4 dequant kernels use on sm_70.
    d_sel = tl.broadcast_to((dim_offsets // 2)[None, :], (BLOCK_N, HEAD_DIM))
    d_sf = tl.broadcast_to((dim_offsets // 16)[None, :], (BLOCK_N, HEAD_DIM))
    d_odd = (dim_offsets % 2 == 1)[None, :]

    n_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    for tile in range(0, n_tiles):
        columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
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
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        # Packed side block per token: [data N*DD][scale N*SD].
        data_base = safe_page * stride_cache_block + page_offset * DD
        scale_base = (
            safe_page * stride_cache_block
            + PAGE_SIZE * DD
            + page_offset * SD
        )

        # K side: dequant [BLOCK_N, D] fp16 in registers, same pattern as
        # the in-tree nvfp4 prefill/dequant kernels.
        raw_k = tl.load(
            k_cache_ptr + data_base[:, None] + tl.arange(0, DD)[None, :],
            mask=valid[:, None],
            other=0,
        )  # [BLOCK_N, DD] u8
        sfraw_k = tl.load(
            k_cache_ptr + scale_base[:, None] + tl.arange(0, SD)[None, :],
            mask=valid[:, None],
            other=0,
        )  # [BLOCK_N, SD] u8
        nib_k = (
            tl.where(d_odd, tl.gather(raw_k, d_sel, 1) >> 4, tl.gather(raw_k, d_sel, 1))
            & 0x0F
        )
        k16 = (
            _nvfp4_e2m1_decode(nib_k) * tl.gather(_nvfp4_e4m3_decode(sfraw_k), d_sf, 1)
        ).to(tl.float16)

        scores = tl.dot(query, tl.trans(k16)) * score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)

        # V side: same dequant form.
        raw_v = tl.load(
            v_cache_ptr + data_base[:, None] + tl.arange(0, DD)[None, :],
            mask=valid[:, None],
            other=0,
        )
        sfraw_v = tl.load(
            v_cache_ptr + scale_base[:, None] + tl.arange(0, SD)[None, :],
            mask=valid[:, None],
            other=0,
        )
        nib_v = (
            tl.where(d_odd, tl.gather(raw_v, d_sel, 1) >> 4, tl.gather(raw_v, d_sel, 1))
            & 0x0F
        )
        v16 = (
            _nvfp4_e2m1_decode(nib_v) * tl.gather(_nvfp4_e4m3_decode(sfraw_v), d_sf, 1)
        ).to(tl.float16)

        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(tl.float16), v16, acc=accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the V layer scale (output_scale, the fp8-arm pattern) into the
    # per-row reciprocal normalizer.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    # Preserve the unfused path's fp16 attention-output rounding before the
    # gate (matches ops/qsa.py's NUM_SPLITS==1 branch).
    normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head_offsets[:, None] * stride_output_gate_head
        + dim_offsets[None, :],
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr
        + row * stride_output_row
        + head_offsets[:, None] * stride_output_head
        + dim_offsets[None, :],
        normalized_output,
        mask=output_mask,
    )


def qsa_nvfp4_shadow_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    output_gate: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    """Run the fp16-splitk QSA skeleton over packed NVFP4 KV (H_kv == 1).

    Args:
        q: fp16 [rows, GROUP_SIZE, HEAD_DIM].
        k_cache: uint8 [num_pages, PAGE, 1, DD+SD] LBHNC side blocks
            (DD = HEAD_DIM//2, SD = HEAD_DIM//16), contiguous blocks.
        v_cache: same layout as k_cache.
        logical_indices: int32 [rows, TOPK+1] packed selection + count.
        block_table: int32 [num_requests, PAGE_TABLE_WIDTH].
        token_to_req: int32 [rows].
        out: fp16 [rows, GROUP_SIZE, HEAD_DIM] (written in place).
        output_gate: fp16 contiguous, view_as(q).
        k_scale: layer float, folded into the softmax scale.
        v_scale: layer float, folded into the output scale.

    """
    rows, group_size, head_dim = q.shape
    page = k_cache.shape[1]
    dd, sd = head_dim // 2, head_dim // 16
    assert k_cache.dtype is torch.uint8 and v_cache.shape == k_cache.shape
    assert k_cache.shape[2] == 1, "H_kv == 1 only in this prototype"
    assert k_cache.shape[3] == dd + sd
    assert k_cache.stride(0) == page * (dd + sd), "contiguous side blocks"
    assert logical_indices.dtype is torch.int32 and block_table.dtype is torch.int32
    assert out.dtype is torch.float16 and out.stride(2) == 1

    topk = logical_indices.shape[1] - 1
    softmax_scale = head_dim**-0.5 * float(k_scale)
    output_scale = float(v_scale)

    block_m = max(4, 2 ** (group_size - 1).bit_length())
    _qsa_nvfp4_splitk_kernel[(rows, 1)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out,
        softmax_scale,
        output_scale,
        output_gate,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        rows,
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=topk,
        PAGE_SIZE=page,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=32,
        # 2026-10-15 knob sweep (sm70_tests/scratch_shadow_tune.py, 4096 rows,
        # 24-config grid): BN=32 warps=4 stages=1 wins (43.7ms vs 60.1ms
        # draft). BN>32 with warps=2 spills registers (~25x); the draft's
        # BN=16/warps=2/stages=2 was 49.1ms.
        num_warps=4,
        num_stages=1,
    )
    return out
