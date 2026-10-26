# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project

"""SM70 (V100) NVFP4 paged PREFILL attention with in-kernel dequantization.

Retires the transient fp16 shadow prefill path (per-step per-layer dequant to
a temporary buffer + fp16 FA2) for ``kv_cache_dtype=nvfp4`` on sm_70. A
flash-style varlen prefill reads the packed NVFP4 paged KV cache directly,
dequantizes K/V in registers, and applies causal + sliding-window masking.
No fp16 KV materialization at all — the O(L^2) shadow dequant traffic of
chunked prefill becomes a single raw-byte KV read per (q tile, layer).

Design notes (V100: 80 SMs, no Triton-visible tensor cores):
- One program per (request, qo head, q tile). Per-qo-head keeps each program's
  tile small (q [BT, D] fp16, acc [BT, D] f32), which V100's 64-register/thread
  budget can hold without spilling; the KV re-read across a GQA group is
  mostly L2-absorbed at short/mid context, and small-batch long-context
  (spec verify) is helped by the extra parallel programs.
- q-tile split: a tile's output rows are independent, so no cross-program
  combine is needed.
- ``tl.dot`` (FMA-lowered on sm_70) keeps the score/acc reductions in
  structured loops — the naive broadcast-sum over D=256 blows up ptxas.

Prefill is always eager (PIECEWISE cudagraph mode keeps prefill out of
graphs), so per-step device metadata (qo_indptr / kv_lens, a few int32s) is
refreshed with a small H2D copy in the metadata builder.

Mask semantics (FA2 ``window_left``-compatible, per query token): the
request's ``q_len`` query tokens are the LAST ``q_len`` tokens of its KV, so
query row ``j`` (0-based within the request) at position
``pos = kv_len - q_len + j`` attends to KV tokens
``[max(0, pos - window_left), pos]`` (window_left < 0 = no window).

Dequant semantics and the LBHNC page layout are identical to
``sm70_nvfp4_dequant.py`` / ``sm70_nvfp4_decode.py`` (bit-exact inverse of
the SM70 store kernel; fp16-rounded K/V, same as the old shadow path).
"""

import torch

from vllm.triton_utils import tl, triton

from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    _nvfp4_e2m1_decode,
    _nvfp4_e4m3_decode,
)

BT_Q: int = 16


@triton.jit
def _sm70_nvfp4_prefill_flash_kernel(
    q_ptr,  # fp16 [total_q, HQ, D]
    kv_ptr,  # uint8 [B, 2H, N, F] (opaque page bytes, LBHNC)
    bt_ptr,  # int32 [R, bt_cols]
    qo_indptr_ptr,  # int32 [R+1] (device): per-request q token ranges
    kvlens_ptr,  # int32 [R] (device): per-request KV lengths
    out_ptr,  # fp16 [total_q, HQ, D]
    k_scale,  # f32 layer _k_scale_float
    v_scale,  # f32 layer _v_scale_float
    sm_scale,  # f32 softmax scale
    page,  # i64 block stride in bytes
    side_off,  # i64 V side base offset from K side base (H * N * F)
    bt_cols,  # i32 block_table row width
    window,  # i32 sliding window_left, < 0 = off
    H: tl.constexpr,  # num kv heads
    HQ: tl.constexpr,  # num qo heads
    D: tl.constexpr,  # head dim
    N: tl.constexpr,  # block size (tokens per page)
    BT: tl.constexpr,  # q tile
):
    """One program per (request, qo head, q tile); flash prefill."""
    DD: tl.constexpr = D // 2  # data bytes per token row
    SD: tl.constexpr = D // 16  # scale bytes per token row
    RATIO: tl.constexpr = HQ // H

    r = tl.program_id(0)
    hq = tl.program_id(1)
    qt = tl.program_id(2)
    h = hq // RATIO

    qo0 = tl.load(qo_indptr_ptr + r)
    q_len = tl.load(qo_indptr_ptr + r + 1) - qo0
    q0 = qt * BT
    if q0 >= q_len:
        return
    kv_len = tl.load(kvlens_ptr + r)

    d = tl.arange(0, D)
    dd = tl.arange(0, DD)
    sd = tl.arange(0, SD)
    n = tl.arange(0, N)
    j = tl.arange(0, BT)
    row_valid = (q0 + j) < q_len

    # Per-row causal + window bounds. Query row j sits at absolute position
    # q_abs0 + q0 + j (q tokens are the last q_len of the request's KV).
    q_abs0 = kv_len - q_len
    pos = q_abs0 + q0 + j  # [BT] absolute query positions
    upper = pos  # causal upper bound (inclusive) per row
    lo_j = tl.where(window >= 0, pos - window, 0)  # window floor per row
    lo_j = tl.maximum(lo_j, 0)
    # lo_j is non-decreasing in j, so the min block floor is the tile's first.
    b_lo = tl.maximum(tl.min(lo_j, axis=0) // N, 0)
    nb_hi = (kv_len + N - 1) // N

    # Query tile: [BT, D] fp16 (invalid rows load 0; their outputs are
    # masked at store time).
    q2 = tl.load(
        q_ptr
        + (qo0 + q0 + j)[:, None].to(tl.int64) * (HQ * D)
        + hq * D
        + d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )  # [BT, D] fp16

    acc = tl.zeros((BT, D), dtype=tl.float32)
    m_i = tl.full((BT,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)

    page64 = page.to(tl.int64)
    h64 = h.to(tl.int64)
    sf_head_base = H * (N * DD) + h64 * (N * SD)
    idx_d = tl.broadcast_to((d // 2)[None, :], (N, D))
    idx_sf = tl.broadcast_to((d // 16)[None, :], (N, D))

    for b in range(b_lo, nb_hi):
        bid = tl.load(bt_ptr + r * bt_cols + b).to(tl.int64)
        data_base = bid * page64 + h64 * (N * DD)
        sf_base = sf_head_base + bid * page64

        # K side: dequant [N, D] fp16 in registers (same pattern as decode).
        raw = tl.load(
            kv_ptr + data_base + n.to(tl.int64)[:, None] * DD + dd[None, :]
        )  # [N, DD] u8
        nib = tl.where(
            (d % 2)[None, :] == 1,
            tl.gather(raw, idx_d, 1) >> 4,
            tl.gather(raw, idx_d, 1),
        ) & 0x0F  # [N, D]
        vals = _nvfp4_e2m1_decode(nib).to(tl.float32)  # [N, D]
        sf = _nvfp4_e4m3_decode(
            tl.load(kv_ptr + sf_base + n.to(tl.int64)[:, None] * SD + sd[None, :])
        ).to(tl.float32) * k_scale  # [N, SD]
        K = (vals * tl.gather(sf, idx_sf, 1)).to(tl.float16)  # [N, D]

        scores = tl.dot(q2, tl.trans(K)) * sm_scale  # [BT, N] f32
        tok = b * N + n
        scores = tl.where(
            (tok[None, :] >= lo_j[:, None]) & (tok[None, :] <= upper[:, None]),
            scores,
            float("-inf"),
        )

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        # Guard: a fully-masked block (all scores -inf) leaves m_new = -inf
        # for a row whose first processed block is below its per-row window
        # floor; exp(-inf - -inf) is NaN. With scores all -inf the softmax
        # weights are 0 either way, so -inf maps to 0 here.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(scores - m_safe[:, None])  # [BT, N]
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V side: dequant [N, D] fp16, accumulate per (q row).
        raw_v = tl.load(
            kv_ptr
            + data_base
            + side_off
            + n.to(tl.int64)[:, None] * DD
            + dd[None, :]
        )  # [N, DD] u8
        nib_v = tl.where(
            (d % 2)[None, :] == 1,
            tl.gather(raw_v, idx_d, 1) >> 4,
            tl.gather(raw_v, idx_d, 1),
        ) & 0x0F  # [N, D]
        vals_v = _nvfp4_e2m1_decode(nib_v).to(tl.float32)  # [N, D]
        sf_v = _nvfp4_e4m3_decode(
            tl.load(
                kv_ptr
                + sf_base
                + side_off
                + n.to(tl.int64)[:, None] * SD
                + sd[None, :]
            )
        ).to(tl.float32) * v_scale  # [N, SD]
        V = (vals_v * tl.gather(sf_v, idx_sf, 1)).to(tl.float16)  # [N, D]

        acc = tl.dot(p.to(tl.float16), V, acc)
        m_i = m_new

    # Normalize and store the tile's rows (invalid rows skipped).
    out = acc / l_i[:, None]
    out = tl.where(l_i[:, None] > 0, out, 0.0)
    tl.store(
        out_ptr
        + (qo0 + q0 + j)[:, None].to(tl.int64) * (HQ * D)
        + hq * D
        + d[None, :],
        out.to(tl.float16),
        mask=row_valid[:, None],
    )


def sm70_nvfp4_paged_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float,
    window_left: int,
    max_q_len: int,
    bt: int = BT_Q,
) -> None:
    """In-kernel-dequant NVFP4 paged PREFILL (varlen, causal, GQA).

    Args:
        q: fp16 ``[total_q, num_qo_heads, D]`` queries (contiguous; prefill
            rows only, in request order).
        kv_cache: raw uint8 layer cache ``[B, 2*H_kv, N, F]`` (LBHNC).
        block_table: int32 ``[num_reqs, max_blocks]`` (device).
        qo_indptr: int32 ``[num_reqs + 1]`` **device** per-request q ranges
            (offsets into ``q`` / ``out``).
        kv_lens: int32 ``[num_reqs]`` **device** per-request KV lengths.
        out: fp16 ``[total_q, num_qo_heads, D]`` (contiguous); written in
            place. Rows without query tokens are untouched.
        k_scale, v_scale: the layer's ``_k_scale_float`` / ``_v_scale_float``.
        sm_scale: softmax scale from the layer.
        window_left: sliding window size in tokens, negative = no window
            (FA2-compatible per-row semantics).
        max_q_len: **host** largest per-request query length (grid sizing;
            avoids a device sync in the hot per-layer path).
        bt: q tile size (power of two, >= 16 required by tl.dot).
    """
    assert q.dtype == torch.float16 and out.dtype == torch.float16
    assert kv_cache.dtype == torch.uint8 and block_table.dtype == torch.int32
    assert qo_indptr.dtype == torch.int32 and kv_lens.dtype == torch.int32
    assert bt >= 16 and (bt & (bt - 1)) == 0
    _, two_h, N, F = kv_cache.shape
    H = two_h // 2
    D = 2 * (F * 8 // 9)
    R = block_table.shape[0]
    HQ = q.shape[1]
    assert HQ % H == 0, f"GQA requires {HQ} qo heads over {H} kv heads"
    assert qo_indptr.shape[0] == R + 1
    q_tiles = max(1, (max_q_len + bt - 1) // bt)
    _sm70_nvfp4_prefill_flash_kernel[(R, HQ, q_tiles)](
        q,
        kv_cache,
        block_table,
        qo_indptr,
        kv_lens,
        out,
        k_scale,
        v_scale,
        sm_scale,
        page=kv_cache.stride(0),
        side_off=H * N * F,
        bt_cols=block_table.shape[1],
        window=window_left,
        H=H,
        HQ=HQ,
        D=D,
        N=N,
        BT=bt,
        num_warps=4,
    )
