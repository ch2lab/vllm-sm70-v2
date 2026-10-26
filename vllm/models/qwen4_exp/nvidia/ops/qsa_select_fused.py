# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D4a shadow: fused QSA decode pre-attn prelude (score + select + expand).

Spec: .superpowers/sdd/2026-10-13-sm70-round4-d4a-launch-fusion/d4a-spec.md
(round-4 candidate D4a, decode slot #1). This module is the shadow kernel of
the decode pre-attention chain: per QSA decode layer, the 3-op prelude
(uniform scoring + ``persistent_topk`` selection + block->token expansion)
plus the caller's ``output.zero_()`` (gated off in ``nvidia/qsa.py``) are
collapsed into ONE Triton launch.

Wiring (all gated by ``VLLM_SM70_QSA_DECODE_FUSE``, default 0 = today):
  * ``QSAIndexer.forward`` (indexer_qsa.py) — pure-decode steps (every row is
    a decode row) route to :func:`qsa_select_fused_decode` instead of
    ``qsa_select_paged_decode`` + ``expand_qsa_block_indices``; mixed batches
    and env=0 run the exact existing chain (byte-identical off path,
    rule 11670). This module is imported lazily inside that gated branch —
    with env=0 it is never imported and never launches.
  * ``forward_qsa`` (nvidia/qsa.py L300) — ``output.zero_()`` is skipped when
    the gate is on (and the batch is non-empty): the sparse attention chain
    fully overwrites ``out`` (the splitk kernel writes 0 for empty rows via
    ``tl.where(has_values, ..., 0.0)`` and the merge kernel writes 0 for
    all-``-inf`` rows; spec §1.2 proof).

Idea-level sources (rule 11671 — recorded here, NOT ported structurally):
  * GDN-layer mirror pattern (spec §1.4): the pre-attention prelude of a
    layer collapses into one fused kernel over static device buffers, with
    no redundant fill and no new persistent state. [in-tree:
    ``qwen_gdn_linear_attn.py`` ``_forward_core_decode_non_spec``]
  * Deterministic lexicographic top-512 selection semantics: score-descending
    / index-ascending, exact k-th-value threshold + tie-broken gather, output
    in index order — the exact-deterministic selector class used by the
    in-tree prefill path (``qsa_indexer.py`` L854-860) and the old-tree T4
    byte-exact evidence. The semantics are those of
    ``csrc/qsa_lexicographic_topk.cuh`` (four radix bytes over ordered float
    bits, exact pivot, first-``remaining`` tie admission, increasing-index
    compaction); re-expressed here as Triton radix passes over the logits
    buffer, not a structural port of the CUDA kernel.
  * 1Cat device-side attention history companion idea (spec §2.1): evaluated
    and REJECTED for D4a v1 — no new persistent state, no graph-state
    lifecycle, zero launch-count benefit.

SM70 notes (V100): one program per decode row; no ldmatrix/TMA assumptions;
smem is the score-pass tile only (~32 KB at SCORE_BLOCK 64 x HEAD_DIM 128
fp16 x STAGES 2 — the same tile as the live ``_qsa_mqa_paged_uniform_kernel``);
the selection and epilogue passes are elementwise over (SEL_BLOCK,) and
(COLUMN_BLOCK,) register vectors. Risk R1 (spec §5): at rows=1 the grid is a
single CTA — occupancy may limit the score pass; the offline gate
(G3 GPU-neutral + online A/B) arbitrates, so this kernel is kept simple and
deliberately un-tuned.

CUDA-graph-capture safety (spec §2.4): reads only the static device buffers
the current in-graph kernels already read (``q``, ``compressed_key_cache``,
``block_table``, ``visible_blocks``, ``logical_positions``); writes the
existing packed selection buffer in place; temporary allocations
(``logits``, ``block_indices``, histogram scratch) are per-step graph-pool
allocations of the same class as today's; no host sync, no ``.item()``,
no shape-dependent allocation (decode shapes are capture-uniform).
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _qsa_ordered_bits(scores):
        # Ordered-float key: monotone in float value, -0.0 and +0.0 in the
        # same bucket (the block index provides the tie break). Semantics of
        # ordered_float_bits, csrc/qsa_lexicographic_topk.cuh L18-24.
        scores = tl.where(scores == 0.0, 0.0, scores)
        bits = scores.to(tl.uint32, bitcast=True)
        neg = (bits & 0x80000000) != 0
        return tl.where(
            neg, ~bits, bits | tl.full(bits.shape, 0x80000000, tl.uint32)
        )

    @triton.jit
    def _qsa_select_fused_kernel(
        q_ptr,
        k_cache_ptr,
        page_table_ptr,
        visible_blocks_ptr,
        logits_ptr,
        block_indices_ptr,
        output_ptr,
        query_positions_ptr,
        hist_ptr,
        stride_q_row,
        stride_q_head,
        stride_cache_block,
        stride_cache_token,
        stride_table_req,
        stride_logits_row,
        stride_out_row,
        stride_hist_row,
        PAGE_SIZE: tl.constexpr,
        PAGE_TABLE_WIDTH: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        OUTPUT_WIDTH: tl.constexpr,
        SCORE_BLOCK: tl.constexpr,
        SEL_BLOCK: tl.constexpr,
        STAGES: tl.constexpr,
        COLUMN_BLOCK: tl.constexpr,
    ) -> None:
        # One program per decode query row: score pass over the row's visible
        # compressed columns, deterministic lexicographic top-BLOCK_TOPK
        # selection, and the packed-buffer expand epilogue.
        NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
        NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
        BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))

        row = tl.program_id(0)
        visible = tl.load(visible_blocks_ptr + row)
        # The csrc reference clamps the live length to the column capacity
        # (qsa_lexicographic_topk.cuh L117-119); visible always <= NUM_COLUMNS
        # for a well-formed page table, the clamp is belt-and-braces.
        length = tl.minimum(visible, NUM_COLUMNS)

        dims = tl.arange(0, BLOCK_D)
        head_offsets = tl.arange(0, NUM_HEADS_PADDED)
        head_valid = head_offsets < NUM_HEADS
        query = tl.load(
            q_ptr
            + row * stride_q_row
            + head_offsets[None, :] * stride_q_head
            + dims[:, None],
            mask=head_valid[None, :] & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )

        # ---- phase 1: uniform scoring. Same tile (SCORE_BLOCK == the live
        # _DECODE_BLOCK_N), same dot, same ReLU + head-sum reduction and same
        # store mask as _qsa_mqa_paged_uniform_kernel (ops/qsa_indexer.py
        # L235-324) -> bit-identical logits for every live column. ----
        column_offsets = tl.arange(0, SCORE_BLOCK)
        tile_end = tl.minimum(
            tl.cdiv(length, SCORE_BLOCK), tl.cdiv(NUM_COLUMNS, SCORE_BLOCK)
        )
        for tile in tl.range(0, tile_end, num_stages=STAGES):
            columns = tile * SCORE_BLOCK + column_offsets
            live = columns < length
            logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
            page_offset = columns % PAGE_SIZE
            physical_page = tl.load(
                page_table_ptr + row * stride_table_req + logical_page,
                mask=live,
                other=0,
            )
            keys = tl.load(
                k_cache_ptr
                + physical_page[:, None].to(tl.int64) * stride_cache_block
                + page_offset[:, None] * stride_cache_token
                + dims[None, :],
                mask=live[:, None] & (dims[None, :] < HEAD_DIM),
                other=0.0,
                eviction_policy="evict_first",
            )
            scores = tl.dot(keys, query, out_dtype=tl.float32)
            scores = tl.where(head_valid[None, :], tl.maximum(scores, 0.0), 0.0)
            score = tl.sum(scores, axis=1)
            tl.store(
                logits_ptr + row * stride_logits_row + columns,
                score,
                mask=(columns < NUM_COLUMNS) & (columns < visible),
            )
        # The phase-1 logits stores must be CTA-wide visible before the
        # phase-2 loads read them (a warp may still be landing tiles while
        # another starts reading; the CUDA reference's separate-kernel
        # structure hides this, here one kernel does both phases).
        tl.debug_barrier()

        # ---- phase 2: deterministic lexicographic top-BLOCK_TOPK selection
        # (score-desc / index-asc; the csrc reference's semantics). The
        # selection is written into block_indices in the reference's
        # canonical order — INCREASING index order over the selected set
        # (all strictly-greater scores first, then the first `remaining`
        # pivot ties; csrc L188-219 / L364-396 "Emit in original index
        # order, matching QSA's canonical accumulation order"). ----
        sel_offsets = tl.arange(0, SEL_BLOCK)
        bin_offsets = tl.arange(0, 256)
        if length <= BLOCK_TOPK:
            # Identity fast path (csrc L123-128): every visible block is
            # selected, in index order; no threshold is needed or possible.
            for c in range(triton.cdiv(BLOCK_TOPK, SEL_BLOCK)):
                idx = c * SEL_BLOCK + sel_offsets
                val = tl.where(idx < length, idx, -1).to(tl.int32)
                tl.store(
                    block_indices_ptr + row * BLOCK_TOPK + idx,
                    val,
                    mask=idx < BLOCK_TOPK,
                )
        else:
            # Pass 1: four radix bytes over the ordered-bit keys of the
            # logits region (pass-1 of the spec's two-pass design: exact
            # k-th-value threshold). Per-row histogram scratch is a (4, 256)
            # int32 region inside the caller's graph-pool buffer.
            prefix = tl.zeros((), dtype=tl.uint32)
            remaining = BLOCK_TOPK
            for p in range(4):
                shift = 24 - p * 8
                hbase = hist_ptr + row * stride_hist_row + p * 256
                tl.store(hbase + bin_offsets, tl.zeros((256,), dtype=tl.int32))
                # The zero-fill must be CTA-wide visible before any same-bin
                # atomic_add (csrc/qsa_lexicographic_topk.cuh L144-148: the
                # histogram clear is sync'd before the atomic loop) — a late
                # zero would drop a count or reuse the previous pass's stale
                # histogram.
                tl.debug_barrier()
                for c in tl.range(0, tl.cdiv(length, SEL_BLOCK)):
                    cols = c * SEL_BLOCK + sel_offsets
                    m = cols < length
                    s = tl.load(
                        logits_ptr + row * stride_logits_row + cols,
                        mask=m,
                        other=0.0,
                    )
                    key = _qsa_ordered_bits(s)
                    if p == 0:
                        mkey = m
                    else:
                        # Fixed high bytes (from prior passes) must match.
                        mkey = m & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
                    tl.atomic_add(hbase + ((key >> shift) & 0xFF), 1, mask=mkey)
                # The atomic updates must complete before the histogram is
                # read by the bin walk (csrc/qsa_lexicographic_topk.cuh
                # L153-160: the atomic loop is sync'd before the walk).
                tl.debug_barrier()
                h = tl.load(hbase + bin_offsets)
                # The reference's walk (qsa_lexicographic_topk.cuh
                # L162-173): running `remaining`, stop at the FIRST (largest)
                # bin b with remaining_b <= h[b], i.e. the largest b with
                # F(b) = sum(h[b..255]) >= remaining (F is non-increasing,
                # so the qualifying bins are the prefix [0, b0]).
                total = tl.sum(h, axis=0)
                F = total - tl.cumsum(h, axis=0) + h
                b0 = (255 - tl.sum((F < remaining).to(tl.int32), axis=0)).to(
                    tl.uint32
                )
                over = tl.sum(h * (bin_offsets.to(tl.uint32) > b0).to(tl.int32), axis=0)
                remaining = remaining - over
                prefix = prefix | (b0 << shift)

            # Pass 2: tie-broken gather (spec's pass-2). All strictly-greater
            # keys are selected; exactly `remaining` pivot ties are admitted
            # in increasing index order; the output position of a selected
            # element is its rank in increasing index order (the csrc offset
            # formula greater_before + min(equal_before, remaining) reduces
            # to that exclusive selected-count — selected elements form a
            # prefix-free interleave, so the exclusive selected count before
            # a column equals greater_before + min(equal_before, remaining)).
            #
            # The tie quota counter must count ADMITTED TIES ONLY (the
            # reference's `equal_before < remaining_ties`, csrc L213/L389):
            # a running selected-rank would wrongly charge the greater
            # elements against the tie quota whenever greater scores and
            # pivot ties interleave across tiles.
            pivot = prefix
            sel_seen = tl.zeros((), dtype=tl.int32)
            eq_seen = tl.zeros((), dtype=tl.int32)
            for c in tl.range(0, tl.cdiv(length, SEL_BLOCK)):
                cols = c * SEL_BLOCK + sel_offsets
                m = cols < length
                s = tl.load(
                    logits_ptr + row * stride_logits_row + cols, mask=m, other=0.0
                )
                key = _qsa_ordered_bits(s)
                greater = m & (key > pivot)
                eq = m & (key == pivot)
                eq_i = eq.to(tl.int32)
                eq_excl = tl.cumsum(eq_i, axis=0) - eq_i
                sel_eq = eq & (eq_seen + eq_excl < remaining)
                selected = greater | sel_eq
                sel_i = selected.to(tl.int32)
                sel_excl = tl.cumsum(sel_i, axis=0) - sel_i
                tl.store(
                    block_indices_ptr + row * BLOCK_TOPK + sel_seen + sel_excl,
                    cols.to(tl.int32),
                    mask=selected,
                )
                sel_seen = sel_seen + tl.sum(sel_i, axis=0)
                eq_seen = eq_seen + tl.sum(sel_eq.to(tl.int32), axis=0)

        # The epilogue below reads the block_indices this program just
        # scattered: make the CTA's global writes visible first (the same
        # barrier idiom as _qsa_prepare_kernel's ring commit).
        tl.debug_barrier()

        # ---- phase 3: packed-buffer epilogue — the exact arithmetic of
        # _expand_qsa_indices_kernel (ops/qsa_indexer.py L438-493): full
        # blocks expanded in selection order, the causal tail of the open
        # group, -1 padding, and the trailing count column. ----
        complete_blocks = tl.minimum(visible, BLOCK_TOPK)
        expanded_count = complete_blocks * COMPRESS_RATIO
        query_position = tl.load(query_positions_ptr + row)
        tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
        tail_count = (query_position + 1) - tail_start
        out_offsets = tl.arange(0, COLUMN_BLOCK)
        for oc in range(triton.cdiv(OUTPUT_WIDTH, COLUMN_BLOCK)):
            cols = oc * COLUMN_BLOCK + out_offsets
            in_width = cols < OUTPUT_WIDTH
            is_expanded = cols < expanded_count
            block_rank = cols // COMPRESS_RATIO
            safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
            block = tl.load(
                block_indices_ptr + row * BLOCK_TOPK + safe_rank,
                mask=is_expanded & in_width,
                other=-1,
            )
            expanded = block * COMPRESS_RATIO + cols % COMPRESS_RATIO
            tail_offset = cols - expanded_count
            is_tail = (
                (cols >= expanded_count)
                & (tail_offset < tail_count)
                & (tail_offset < COMPRESS_RATIO - 1)
            )
            token = tl.where(is_expanded, expanded, tail_start + tail_offset)
            valid = in_width & (is_expanded | is_tail) & (token >= 0)
            tl.store(
                output_ptr + row * stride_out_row + cols,
                tl.where(valid, token, -1).to(tl.int32),
                mask=in_width,
            )
        # TRAILING COUNT COLUMN: the packed buffer is OUTPUT_WIDTH+1 wide;
        # the last column holds this row's valid-entry count (expanded blocks
        # plus causal tail). It is NOT a token index — the sparse attention
        # kernel reads it as the row's tile-loop bound (same as the expand
        # kernel's program_id(1)==0 store).
        tl.store(
            output_ptr + row * stride_out_row + OUTPUT_WIDTH,
            (expanded_count + tail_count).to(tl.int32),
        )


def qsa_select_fused_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    query_positions: torch.Tensor,
    out: torch.Tensor,
    *,
    logits_out: torch.Tensor | None = None,
) -> None:
    """One-launch fused prelude: uniform scoring + lexicographic top-k +
    packed-buffer expansion, for a request-major pure-decode batch.

    The input arg set mirrors the current 3-op chain (``qsa_indexer.py``
    L750-818: ``q``/``k_cache``/``page_table``/``visible_blocks``/
    ``token_topk``/``compress_ratio``/``decode_query_len``); ``query_positions``
    and ``out`` are added because the absorbed expand op (#5 of the spec's
    chain) consumes them — it writes the PACKED selection buffer directly
    (no separate block_indices hand-off across launches).

    Args follow ``qsa_select_paged_decode`` plus:
        query_positions: per-row logical token position (int64, the
            ``logical_positions`` metadata) — the expand tail's input.
        out: packed selection buffer ``[rows, token_topk + compress_ratio]``
            int32, written in place (the attention kernel's input).

    Temporary graph-pool allocations (same class as today's off-path
    per-step temporaries; no new persistent state, VRAM delta 0 by
    construction — still measured, rule 11672): the fp32 ``logits`` row
    buffer, the intermediate ``block_indices`` (rows x block_topk), and a
    per-row (4 x 256) int32 radix-histogram scratch.

    ``logits_out`` (bench hook only, default None): a caller-owned fp32
    ``[rows, columns]`` buffer the score pass writes instead of an
    internal allocation — the gate-1 parity check (spec §3.1 G1a) reads
    it back. The model path never passes it (internal allocation, as
    today).
    """
    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_topk == 512, (
        "D4a v2 fuses the live k=512 selector (spec §2.1); other widths "
        "would need the ladder-rung-2 form"
    )
    assert q.ndim == 3 and q.shape[0] > 0
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"
    assert k_cache.ndim == 4 and k_cache.shape[3] == q.shape[2]
    assert page_table.ndim == 2
    assert visible_blocks.shape == (q.shape[0],)
    assert query_positions.shape == (q.shape[0],)
    assert out.shape == (q.shape[0], output_width + 1)
    assert out.dtype == torch.int32 and out.stride(1) == 1
    assert visible_blocks.is_contiguous()

    rows = q.shape[0]
    columns = page_table.shape[1] * k_cache.shape[1]
    if logits_out is not None:
        logits = logits_out
        assert logits.shape == (rows, columns) and logits.dtype == torch.float32
        assert logits.stride(1) == 1
    else:
        logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    block_indices = torch.empty((rows, block_topk), dtype=torch.int32, device=q.device)
    hist = torch.empty((rows, 4, 256), dtype=torch.int32, device=q.device)
    # One program per decode query row (rows = decode_query_len x num_requests
    # for the uniform decode batch; the graph's decode shape is rows=1).
    _qsa_select_fused_kernel[(rows,)](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        block_indices,
        out,
        query_positions,
        hist,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        out.stride(0),
        hist.stride(0),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        OUTPUT_WIDTH=output_width,
        # Same score tile as the live _DECODE_BLOCK_N for bit-identical logits.
        # R1-parallelism A/B knobs (defaults keep the bit-identical contract):
        # changing them alters the tl.dot tile/thread/fragment mapping, so the
        # logits are no longer bit-identical to the live selector — a changed
        # tile must be re-validated with the selection-parity + content gates.
        SCORE_BLOCK=int(os.environ.get("VLLM_SM70_QSA_FUSE_SCORE_BLOCK", "64")),
        SEL_BLOCK=128,
        STAGES=int(os.environ.get("VLLM_SM70_QSA_FUSE_STAGES", "2")),
        COLUMN_BLOCK=256,
        # num_warps=2: match the live _qsa_mqa_paged_uniform_kernel launch
        # (num_warps=2 for the non-fp8 cache, qsa_indexer.py L809) so the
        # score pass's tl.dot lowering (tile/thread/fragment mapping) is
        # identical and the logits stay bit-identical (spec §2.1, gate G1a).
        # The elementwise selection/epilogue passes are simple at 2 warps.
        num_warps=int(os.environ.get("VLLM_SM70_QSA_FUSE_WARPS", "2")),
    )


__all__ = ["qsa_select_fused_decode"]
