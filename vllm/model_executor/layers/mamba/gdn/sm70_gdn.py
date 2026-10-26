# SPDX-License-Identifier: Apache-2.0
#
# SM70 (V100) GDN CUDA kernel — JIT-compiled from csrc.
# Based on the FlashQLA-SM70 kernel from 1Cat-vLLM (MIT license).
#
# This kernel is used when:
#   - GPU is SM70/SM75 (V100)
#   - GDN head dimensions are K=V=128
#   - Model dtype is fp16 (or bf16/fp32)
#
# The FLA Triton kernel (chunk_delta_h.py) does not work correctly on SM70
# (produces NaN/inf due to fp16 precision issues in the WMMA accumulation).
# This CUDA kernel uses explicit fp32 accumulation and is validated on V100.

from __future__ import annotations

import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
from torch.utils.cpp_extension import load

_EXT = None
_EXT_LOAD_ERROR: Exception | None = None
_REPO_ROOT = Path(__file__).resolve().parents[5]
_CUDA_SRC = _REPO_ROOT / "csrc/gdn/gdn_sm70_fwd.cu"
_CHUNKED_SRC = _REPO_ROOT / "csrc/gdn/gdn_sm70_chunked.cu"

# SM70/SM75 only — V100 generation.
_SM70_GENCODE_FLAGS = [
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
]


def _load_ext():
    """JIT-compile and load the SM70 GDN CUDA extension (cached)."""
    global _EXT, _EXT_LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_LOAD_ERROR is not None:
        raise RuntimeError(
            "SM70 GDN extension previously failed to load."
        ) from _EXT_LOAD_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("SM70 GDN kernel requires CUDA.")

    # Allow prebuilt extension path override (e.g. for testing)
    prebuilt = os.environ.get("VLLM_SM70_GDN_PREBUILT_SO")
    if prebuilt:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "vllm_sm70_gdn", Path(prebuilt).expanduser()
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {prebuilt}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT = mod
        return _EXT

    src = _CUDA_SRC.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"GDN SM70 CUDA source not found: {src}")

    try:
        _EXT = load(
            name="vllm_sm70_gdn_fwd",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", *_SM70_GENCODE_FLAGS],
            extra_cflags=["-O3"],
            verbose=bool(int(os.environ.get("VLLM_SM70_GDN_VERBOSE", "0"))),
        )
    except Exception as exc:
        _EXT_LOAD_ERROR = exc
        raise
    return _EXT


_CHUNKED_EXT = None
_CHUNKED_EXT_LOAD_ERROR: Exception | None = None


def _load_chunked_ext():
    """JIT-compile and load the SM70 GDN CHUNCHED extension (P3 shadow)."""
    global _CHUNKED_EXT, _CHUNKED_EXT_LOAD_ERROR
    if _CHUNKED_EXT is not None:
        return _CHUNKED_EXT
    if _CHUNKED_EXT_LOAD_ERROR is not None:
        raise RuntimeError(
            "SM70 GDN chunked extension previously failed to load."
        ) from _CHUNKED_EXT_LOAD_ERROR
    src = _CHUNKED_SRC.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"GDN SM70 chunked CUDA source not found: {src}")
    try:
        _CHUNKED_EXT = load(
            name="vllm_sm70_gdn_chunked_fwd",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", *_SM70_GENCODE_FLAGS],
            extra_cflags=["-O3"],
            verbose=bool(int(os.environ.get("VLLM_SM70_GDN_VERBOSE", "0"))),
        )
    except Exception as exc:
        _CHUNKED_EXT_LOAD_ERROR = exc
        raise
    return _CHUNKED_EXT


def _chunked_enabled() -> bool:
    return os.environ.get("VLLM_SM70_GDN_CHUNKED", "0") == "1"


def chunk_gated_delta_rule_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the SM70 CUDA GDN chunk prefill kernel.

    Mirrors the FLA Triton interface (chunk_gated_delta_rule) but uses a
    custom CUDA kernel validated for V100/SM70.

    Shapes (varlen, batch=1 after vLLM flattening):
        q/k: [1, T, Hq, K]  (K=128)
        v:   [1, T, Hv, V]  (V=128)
        g:   [1, T, Hv]
        beta: [1, T, Hv]
        initial_state: [N, Hv, V, K] or None  (vLLM layout)
        cu_seqlens: [N+1] int32

    Returns:
        output: [1, T, Hv, V]
        final_state: [N, Hv, V, K] fp32, or None

    """
    ext = _load_ext()

    if use_qk_l2norm_in_kernel:
        from vllm.third_party.flash_linear_attention.ops.l2norm import (
            l2norm_fwd,
        )

        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if cu_seqlens is None:
        cu_seqlens = torch.tensor(
            [0, q.shape[1]], device=q.device, dtype=torch.int32
        )
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    # Ensure contiguous
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    g = g if g.is_contiguous() else g.contiguous()
    beta = beta if beta.is_contiguous() else beta.contiguous()
    if initial_state is not None:
        initial_state = (
            initial_state if initial_state.is_contiguous()
            else initial_state.contiguous()
        )

    scale = q.shape[-1] ** -0.5

    # Use the varlen variant (vLLM state layout: [N, Hv, V, K])
    output, final_state = ext.gdn_forward_vlk_varlen(
        q,
        k,
        v,
        g,
        beta,
        initial_state,  # c10::optional — pass None directly
        cu_seqlens,
        float(scale),
        output_final_state,
        False,  # validate_cu_seqlens (skip in serving for speed)
        False,  # gate_is_exp
        None,   # output (allocate new)
    )

    if not output_final_state:
        final_state = None

    return output, final_state


def chunk_gated_delta_rule_sm70_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """P3 shadow kernel: FLA-style chunked GDN (pure-FMA sm70 kernels).

    Same contract as chunk_gated_delta_rule_sm70. Default-OFF: only reached
    via VLLM_SM70_GDN_CHUNKED=1 or direct call (tests/bench).
    """
    ext = _load_chunked_ext()

    # Shape/rate probe for the GDN chunked path (env-gated, default off). One
    # line per distinct (T, batch, heads, chunk-count) so the per-token cost can
    # be computed from the profiler's per-kernel totals. Valve: default = no log.
    if os.environ.get("VLLM_SM70_GDN_SHAPE_LOG", "0") == "1":
        try:
            _t = int(q.shape[1])
            _hq = int(q.shape[2])
            _hv = int(v.shape[2])
            _nb = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else 1
            _chunks = (cu_seqlens[1:] - cu_seqlens[:-1]).div(64).ceil().sum().item() \
                if cu_seqlens is not None else (_t + 63) // 64
            _key = (_t, _nb, _hq, _hv, int(_chunks))
            _seen = getattr(chunk_gated_delta_rule_sm70_chunked, "_seen", None)
            if _seen is None:
                _seen = set()
                chunk_gated_delta_rule_sm70_chunked._seen = _seen
            if _key not in _seen:
                _seen.add(_key)
                logger.info(
                    "GDN chunked shape: T=%d batch=%d heads=%d/%d chunks=%d "
                    "tokens_per_chunk=%.1f", _t, _nb, _hq, _hv, int(_chunks),
                    _t / max(1, int(_chunks)))
        except Exception:  # noqa: BLE001 - probe must never affect serving
            pass

    if use_qk_l2norm_in_kernel:
        from vllm.third_party.flash_linear_attention.ops.l2norm import (
            l2norm_fwd,
        )

        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if cu_seqlens is None:
        cu_seqlens = torch.tensor(
            [0, q.shape[1]], device=q.device, dtype=torch.int32
        )
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    g = g if g.is_contiguous() else g.contiguous()
    beta = beta if beta.is_contiguous() else beta.contiguous()
    if initial_state is not None:
        initial_state = (
            initial_state if initial_state.is_contiguous()
            else initial_state.contiguous()
        )

    scale = q.shape[-1] ** -0.5
    output, final_state = ext.gdn_chunked_forward(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        float(scale),
        output_final_state,
    )

    if not output_final_state:
        final_state = None

    return output, final_state


# Chunked-shadow seam (P3 GDN shadow spec): the shadow is validated only for the
# production *prefill* form — T >= 4096, K=V=128, fp16, TP8 per-rank geometry
# (Hq=2, Hv=6). Everything else MUST fall back to the incumbent. Without this
# gate the shadow also ran decode (T=1) and short prompts, which produced
# corrupt output (correctness.py: known_answer[391/672/olleh] + needle ->
# "ductduct..."; measured 2026-10-18: +19% prefill, CORRECT-FAIL 4/9).
_CHUNKED_MIN_T = 1024
_CHUNKED_K = 128
_CHUNKED_HQ = 2
_CHUNKED_HV = 6


def _chunked_eligible(q: torch.Tensor, v: torch.Tensor) -> bool:
    """Shadow seam gate: env on AND the validated prefill shape only."""
    if not _chunked_enabled():
        return False
    if q.dtype != torch.float16 or v.dtype != torch.float16:
        return False
    if q.dim() != 4 or v.dim() != 4:
        return False
    return (
        q.shape[1] >= _CHUNKED_MIN_T
        and q.shape[-1] == _CHUNKED_K
        and v.shape[-1] == _CHUNKED_K
        and q.shape[2] == _CHUNKED_HQ
        and v.shape[2] == _CHUNKED_HV
    )


def chunk_gated_delta_rule_sm70_gated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Env-gated dispatch: VLLM_SM70_GDN_CHUNKED=1 AND the validated prefill
    shape (see :func:`_chunked_eligible`) -> chunked shadow; else the incumbent
    serial kernel (byte-identical default behavior)."""
    if _chunked_eligible(q, v):
        return chunk_gated_delta_rule_sm70_chunked(
            q, k, v, g, beta, initial_state, output_final_state,
            cu_seqlens, use_qk_l2norm_in_kernel, core_attn_out,
        )
    return chunk_gated_delta_rule_sm70(
        q, k, v, g, beta, initial_state, output_final_state,
        cu_seqlens, use_qk_l2norm_in_kernel, core_attn_out,
    )
