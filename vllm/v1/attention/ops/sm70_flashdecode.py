# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 (V100) unified NVFP4 paged decode/verify — FlashDecoding kernel op.

Split-KV decode with a fixed grid (cudagraph-safe) that reads the packed
NVFP4 paged KV cache directly, dequantizes K/V in registers (no gather-based
codegen — the sm_70 Triton miscompilation that caps the legacy
``sm70_nvfp4_paged_decode`` at ~1.8 GB/s effective KV bandwidth), and runs
dim-lane SIMD attention (one qo-head group per kv head; 1x KV traffic). Two kernels:

- partition kernel: grid = (splits, H_kv, R); each block handles one kv head's
  qo-head group over a contiguous token span, emitting a partial
  online-softmax state (acc | m | l) per (q row, qo head).
- reduce kernel: grid = (H_kv, R); merges the split partials
  (log-sum-exp combine) into the final output.

q_rows <= 8: q_rows=1 is pure decode; q_rows>1 is speculative verify where
row ``qq`` sits at sequence position ``kv_len - q_rows + qq`` and attends
causally to ``[lo, kv_len - q_rows + qq + 1)``.

LBHNC byte layout (one raw page, identical to ``sm70_nvfp4_dequant.py``):

    data  (K side, head h, token n, dim d):
        b*page + h*(N*DD) + n*DD + d//2          (DD = D//2 bytes)
    scale (K side):
        H*N*DD + b*page + h*(N*SD) + n*SD + d//16 (SD = D//16 bytes)
    V side = same + side_off (H*N*F, F = DD + SD)

Nibble packing: even dim = low nibble of byte d//2, odd = high nibble.
Dequant: x_hat = E2M1[nib] * E4M3[sf_byte] * layer_scale.
"""

import numpy as np
import torch

from vllm import envs

from vllm.v1.attention.ops.sm70_nvfp4_dequant import (
    _nvfp4_e2m1_decode,
    _nvfp4_e4m3_decode,
)


def _e2m1_vec(x: np.ndarray) -> np.ndarray:
    """e2m1 nibble -> float32 (vectorized). Mirrors _nvfp4_e2m1_decode."""
    x = np.asarray(x, dtype=np.uint8)
    mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    v = mag[x & 7]
    return np.where(x & 8, -v, v)


def _e4m3_vec(x: np.ndarray) -> np.ndarray:
    """e4m3 scale byte -> float32 (vectorized). Mirrors _nvfp4_e4m3_decode."""
    x = np.asarray(x, dtype=np.uint8)
    sgn = (x & 0x80).astype(np.bool_)
    e = ((x >> 3) & 0xF).astype(np.int32)
    m = (x & 7).astype(np.float32)
    ef = e.astype(np.float32)
    val = np.where(e == 0, m * 0.001953125, (1.0 + m * 0.125) * np.exp2(ef - 7.0))
    return np.where(sgn, -val, val)


def sm70_nvfp4_reference_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float,
    window_left: int,
) -> torch.Tensor:
    """Exact CPU reference for the unified flashdecode op (numpy).

    Args:
        q: fp16 ``[R, Q, HQ, D]`` (Q = q_rows; row ``qq`` causally attends to
            ``[lo, kv_len - Q + qq + 1)``).
        kv_cache: uint8 ``[B, 2H, N, F]`` raw LBHNC bytes (F = D/2 + D/16).
        block_table: int32 ``[R, max_blocks]`` (CPU or device).
        kv_lens: int32 ``[R]`` (CPU or device).
        k_scale, v_scale: layer ``_k_scale_float`` / ``_v_scale_float``.
        sm_scale: softmax scale.
        window_left: sliding window in tokens, < 0 = off.

    Returns:
        fp32 ``[R, Q, HQ, D]``.
    """
    qf = q.detach().cpu().float().numpy()
    kv = kv_cache.detach().cpu().view(torch.uint8).flatten().numpy()
    bt = block_table.detach().cpu().numpy()
    kl = kv_lens.detach().cpu().numpy()
    B, two_H, N, F = kv_cache.shape
    H = two_H // 2
    R, Q, HQ, D = qf.shape
    RATIO = HQ // H
    S0 = kv_cache.stride(0)  # page stride in bytes
    DD, SD = D // 2, D // 16
    side_off = H * N * F
    scale_off = H * N * DD

    out = np.zeros((R, Q, HQ, D), dtype=np.float32)
    for r in range(R):
        kv_len = int(kl[r])
        if kv_len == 0:
            continue
        lo = kv_len - 1 - window_left if window_left >= 0 else 0
        lo = max(lo, 0)
        if lo >= kv_len:
            continue
        # Precompute dequantized K/V for this request (per kv head).
        KV = []
        for h in range(H):
            toks = np.arange(lo, kv_len, dtype=np.int64)
            pages = bt[r, toks // N].astype(np.int64)
            rows = (toks % N).astype(np.int64)
            K = np.zeros((len(toks), D), dtype=np.float32)
            V = np.zeros_like(K)
            for side, scale in ((0, k_scale), (side_off, v_scale)):
                # Data: [ntok, DD] bytes -> nibbles [ntok, D]
                db = pages * S0 + side + h * (N * DD) + rows * DD
                data = kv[db[:, None] + np.arange(DD)[None, :]].astype(np.uint8)
                nib = np.empty((len(toks), D), dtype=np.uint8)
                nib[:, 0::2] = data & 0x0F
                nib[:, 1::2] = data >> 4
                # Scales: [ntok, SD] bytes -> [ntok, D]
                sb = (
                    scale_off
                    + pages * S0
                    + side
                    + h * (N * SD)
                    + rows * SD
                )
                sf = kv[sb[:, None] + np.arange(SD)[None, :]].astype(np.uint8)
                sf_b = _e4m3_vec(sf)  # [ntok, SD]
                val = _e2m1_vec(nib) * np.repeat(sf_b, 16, axis=1) * scale
                if side == 0:
                    K[:] = val
                else:
                    V[:] = val
            KV.append((K, V))
        # Attention per (q row, qo head): causal hi per row.
        for qq in range(Q):
            hi = min(kv_len - Q + qq + 1, kv_len)
            if hi <= lo:
                continue
            for hq in range(HQ):
                h = hq // RATIO
                Kq = KV[h][0][: hi - lo]
                Vq = KV[h][1][: hi - lo]
                s = Kq @ qf[r, qq, hq]
                s = (s * sm_scale).astype(np.float32)
                p = np.exp(s - s.max())
                out[r, qq, hq] = (p @ Vq) / p.sum()
    return torch.from_numpy(out)


def sm70_flashdecode_op(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float,
    window_left: int,
    splits: int,
    q_rows: int,
    workspace: torch.Tensor,
) -> None:
    """Launch the unified NVFP4 flashdecode op (writes ``out`` in place).

    Args:
        q: fp16 ``[R, Q, HQ, D]`` contiguous; Q == q_rows (padded).
        kv_cache: uint8 ``[B, 2H, N, F]`` LBHNC raw bytes (device).
        block_table: int32 ``[R, max_blocks]`` (device).
        kv_lens: int32 ``[R]`` (device).
        out: fp16 ``[R, Q, HQ, D]`` (device); written in place.
        k_scale, v_scale: layer ``_k_scale_float`` / ``_v_scale_float``.
        sm_scale: softmax scale.
        window_left: sliding window, < 0 = off.
        splits: KV split count (grid dim; fixed per launch / capture).
        q_rows: active rows per request (1 = decode, <= 8 = verify).
        workspace: fp32 with >=
            ``R * H * splits * q_rows * (HQ // H) * (D + 2)``
            + ``4 * R * H * q_rows * (HQ // H) * (D + 2)`` elements
            (persistent; reused across layers/steps).
    """
    assert q.dtype == torch.float16 and out.dtype == torch.float16
    assert kv_cache.dtype == torch.uint8 and block_table.dtype == torch.int32
    assert (
        q.device == kv_cache.device == block_table.device
        == kv_lens.device == out.device == workspace.device
        and q.device.type == "cuda"
    ), "sm70_flashdecode: all tensors must be on the same cuda device"
    R, Q, HQ, D = q.shape
    assert Q == q_rows
    H = kv_cache.shape[1] // 2
    assert HQ % H == 0, f"GQA requires {HQ} qo heads over {H} kv heads"
    assert (
        R * H * splits * q_rows * (HQ // H) * (D + 2)
        + 4 * R * H * q_rows * (HQ // H) * (D + 2)
        <= workspace.numel()
    ), "flashdecode workspace too small"
    try:
        op = (
            torch.ops._C.sm70_flashdecode_hmma
            if envs.VLLM_SM70_FD_HMMA
            else torch.ops._C.sm70_flashdecode
        )
    except AttributeError:
        raise RuntimeError(
            "sm70_flashdecode kernel not built — "
            "add csrc/libtorch_stable/attention/sm70_flashdecode.cu and run "
            "build_sm70.sh (plan Task 2)"
        )
    op(
        q,
        kv_cache,
        block_table,
        kv_lens,
        out,
        float(k_scale),
        float(v_scale),
        float(sm_scale),
        int(window_left),
        int(splits),
        int(q_rows),
        workspace,
    )


def sm70_grouped_verify_op(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float,
    q_rows: int,
    workspace: torch.Tensor,
) -> None:
    """Launch the grouped-verify op (1Cat port, 80-split one-wave; writes
    ``out`` in place).

    Same tensor contract as :func:`sm70_flashdecode_op` except there is no
    ``window_left``/``splits`` (the split count is the kernel's fixed 80)
    and ``q_rows`` must be 8 (the DFlash2 verify shape).

    Args:
        workspace: fp32 with >= ``R * 80 * q_rows * HQ * 384`` elements
            (R*80*q_rows*HQ*D fp16 partials + R*80*q_rows*HQ*D fp32 LSE).
    """
    assert 1 <= q_rows <= 8, (
        "sm70_grouped_verify: q_rows must be in [1, 8] (MTP/DFlash verify "
        "rows); the kernel Q panel is sized for 8")
    assert q.dtype == torch.float16 and out.dtype == torch.float16
    assert kv_cache.dtype == torch.uint8 and block_table.dtype == torch.int32
    assert (
        q.device == kv_cache.device == block_table.device
        == kv_lens.device == out.device == workspace.device
        and q.device.type == "cuda"
    ), "sm70_grouped_verify: all tensors must be on the same cuda device"
    R, Q, HQ, D = q.shape
    assert Q == q_rows
    H = kv_cache.shape[1] // 2
    assert HQ % H == 0, f"GQA requires {HQ} qo heads over {H} kv heads"
    assert (
        R * 80 * q_rows * HQ * 384 <= workspace.numel()
    ), "grouped verify workspace too small"
    torch.ops._C.sm70_grouped_verify_nvfp4(
        q,
        kv_cache,
        block_table,
        kv_lens,
        out,
        float(k_scale),
        float(v_scale),
        float(sm_scale),
        int(q_rows),
        workspace,
    )
