# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the SM70 QSA prefill CUDA/WMMA extension (P1, T2.1).

New file, isolated: no change to the system main flow. Imported lazily and
only drives the new csrc kernel for offline gates (design/subplan
2026-10-14-p1-qsa-prefill-kernel-{spec,subplan}.md).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_SM70_GENCODE_FLAGS = [
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
]

_EXT = None
_ERR = None


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "csrc").is_dir():
            return parent
    raise RuntimeError("repo root (containing csrc/) not found")


def _cuda_src() -> Path:
    return _repo_root() / "csrc" / "qsa" / "qsa_prefill_sm70.cu"


def _load_ext():
    global _EXT, _ERR
    if _EXT is not None:
        return _EXT
    if _ERR is not None:
        raise RuntimeError(
            "SM70 QSA prefill extension previously failed to load."
        ) from _ERR
    if not torch.cuda.is_available():
        raise RuntimeError("SM70 QSA prefill kernel requires CUDA.")
    src = _cuda_src().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"CUDA source not found: {src}")
    try:
        _EXT = load(
            name="vllm_sm70_qsa_prefill",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", *_SM70_GENCODE_FLAGS],
            extra_cflags=["-O3"],
            verbose=bool(int(os.environ.get("VLLM_SM70_QSA_PREFILL_VERBOSE", "0"))),
        )
    except Exception as exc:
        _ERR = exc
        raise
    return _EXT


def wmma_qk_probe(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """S = scale * Q @ K^T ; q [16,256] fp16, k [64,256] fp16 -> [16,64] fp32."""
    return _load_ext().qsa_wmma_qk_probe(q, k, scale)


def wmma_attn_probe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    scale: float) -> torch.Tensor:
    """O = softmax(scale*Q@K^T) @ V ; q[16,256], k/v[64,256] fp16 -> [16,256] fp16."""
    return _load_ext().qsa_wmma_attn_probe(q, k, v, scale)


def prefill_sm70_forward(q: torch.Tensor, k_cache: torch.Tensor,
                         v_cache: torch.Tensor, indices: torch.Tensor,
                         block_table: torch.Tensor, token_to_req: torch.Tensor,
                         out: torch.Tensor, gate: torch.Tensor,
                         topk: int, scale: float) -> torch.Tensor:
    """Paged sparse QSA prefill attention (T2.2, WMMA): one CTA per query row."""
    return _load_ext().qsa_prefill_sm70_forward(
        q, k_cache, v_cache, indices, block_table, token_to_req, out, gate,
        topk, scale)


def prefill_fma_forward(q: torch.Tensor, k_cache: torch.Tensor,
                        v_cache: torch.Tensor, indices: torch.Tensor,
                        block_table: torch.Tensor, token_to_req: torch.Tensor,
                        out: torch.Tensor, gate: torch.Tensor,
                        topk: int, scale: float) -> torch.Tensor:
    """Paged sparse QSA prefill attention (T2.3b, FMA-layout): one CTA per row."""
    # The FMA kernel addresses the KV caches flat (contiguous). The live KV
    # pools are strided, non-contiguous views of the full cache tensor, so a
    # strided input makes flat reads hit the wrong addresses (0/NaN outputs).
    if not k_cache.is_contiguous():
        k_cache = k_cache.contiguous()
    if not v_cache.is_contiguous():
        v_cache = v_cache.contiguous()
    return _load_ext().qsa_prefill_fma_forward(
        q, k_cache, v_cache, indices, block_table, token_to_req, out, gate,
        topk, scale)


def prefill_fma_phase(q: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, indices: torch.Tensor,
                      block_table: torch.Tensor, token_to_req: torch.Tensor,
                      out: torch.Tensor, gate: torch.Tensor, topk: int,
                      scale: float, skip: int) -> torch.Tensor:
    """T10 phase-split FMA attribution (bench-only): skip bitmask 1=gather,
    2=scores, 4=PV. Not on the serving path."""
    if not k_cache.is_contiguous():
        k_cache = k_cache.contiguous()
    if not v_cache.is_contiguous():
        v_cache = v_cache.contiguous()
    return _load_ext().qsa_prefill_fma_phase(
        q, k_cache, v_cache, indices, block_table, token_to_req, out, gate,
        topk, scale, skip)
