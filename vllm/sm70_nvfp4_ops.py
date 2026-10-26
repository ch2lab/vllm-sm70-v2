# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 (Volta / V100) NVFP4 op bindings.

Binds the SM70 NVFP4 ops (registered in the vLLM C extension). Enabled
automatically on exact-SM70 hardware (is_enabled = hardware check + registered
op); there is no env toggle -- this tree is SM70-production-specific and the
hardware check fails closed everywhere else.

The 1x storage layout is the QPN2 4-bit column-major layout (one single
4-bit weight per linear layer). Decode (m<=32) reads it directly via the
QPN2 high-occupancy GEMM; prefill (m>32) dequants it to transient FP16 and
runs a plain cuBLAS GEMM.
"""

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["is_exact_sm70", "is_enabled", "dequant", "dequant_from_qpn2",
           "qpn2_prepare", "qpn2_gemm", "gemm_dispatch", "qpn2_prepare_moe",
           "qpn2_moe_gemm", "qpn2_moe_enabled"]


def is_exact_sm70() -> bool:
    """Return True iff the current CUDA device is exactly SM70 (V100/Volta)."""
    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability()) == (7, 0)


def _op_registered() -> bool:
    """True iff the SM70 NVFP4 ops are present in the loaded C extension."""
    try:
        return hasattr(torch.ops._C, "sm70_nvfp4_dequant")
    except Exception:
        return False


def is_enabled() -> bool:
    """Return True iff the SM70 NVFP4 path is usable."""
    if not is_exact_sm70():
        return False
    return _op_registered()


def qpn2_prepare(weight, weight_scale):
    """Prepack the checkpoint row-major NVFP4 weight (n, k//2) uint8 and
    per-16-group E4M3 scales (n, k//16) into the QPN2 column-major layout
    (one-time, at weight-load). Returns (codes, scales), both uint8."""
    return torch.ops._C.sm70_nvfp4_qpn2_prepare(
        weight.contiguous(), weight_scale.contiguous())


def qpn2_gemm(out, input, codes, scales, global_scale, split_k=8, nacc=1):
    """SM70 NVFP4 QPN2 high-occupancy decode GEMM: out = input @ W^T.

    For m <= 32 (decode). global_scale is the layer raw weight_global_scale.
    """
    if torch.is_tensor(global_scale):
        global_scale = float(global_scale.item())
    torch.ops._C.sm70_nvfp4_qpn2_gemm(
        out, input, codes, scales, float(global_scale),
        int(split_k), int(nacc))


def dequant(weight, weight_scale, global_scale):
    """Dequant natural NVFP4 weight (n, k//2) uint8 + E4M3 scales to (n,k)
    FP16. Single fused SM70 C++ kernel (E2/M1 4-LUT + fp16 math)."""
    if torch.is_tensor(global_scale):
        global_scale = float(global_scale.item())
    return torch.ops._C.sm70_nvfp4_dequant(
        weight.contiguous(), weight_scale.contiguous(), float(global_scale))


def dequant_from_qpn2(codes, scales, global_scale):
    """Dequant QPN2 4-bit layout (n,k//2) codes + (n,k//16) scales to
    natural (n,k) FP16 weight. Same result as dequant(raw_w, raw_s, gs)."""
    if torch.is_tensor(global_scale):
        global_scale = float(global_scale.item())
    return torch.ops._C.sm70_nvfp4_dequant_from_qpn2(
        codes, scales, float(global_scale))


def gemm_dispatch(out, input, qpn2_codes, qpn2_scales, qpn2_global_scale,
                  split_k=8, nacc=1):
    """Single-op dispatch: QPN2 for m<=32, dequant+cuBLAS for m>32.
    The m-based decision is inside C++ (concrete int)."""
    if torch.is_tensor(qpn2_global_scale):
        qpn2_global_scale = float(qpn2_global_scale.item())
    torch.ops._C.sm70_nvfp4_gemm_dispatch(
        out, input, qpn2_codes, qpn2_scales, float(qpn2_global_scale),
        int(split_k), int(nacc))


def qpn2_prepare_moe(weight, weight_scale):
    """Batched one-time (at weight-load) prepack of natural-layout expert
    weights [E, n, k//2] uint8 + [E, n, k//16] E4M3 group scales into the QPN2
    lane-major layout. Returns (codes, scales), both uint8, same shapes."""
    return torch.ops._C.sm70_nvfp4_qpn2_prepare_moe(
        weight.contiguous(), weight_scale.contiguous())


def qpn2_moe_gemm(out, input, codes, scales, global_scale, sorted_token_ids,
                  expert_ids, num_tokens_past_padded, topk_weights, top_k,
                  mul_topk_weights=False, num_shards=1,
                  moe_block_size=16, fuse_act=False):
    """Route-tiled (grouped) NVFP4 QPN2 expert GEMM: one launch computes every
    expert that this step routes to, with a CTA per (32 output columns,
    moe_align_block_size route block).

    out is the expanded-token buffer [m*top_k, n] (intermediate_cache1 / 2
    layout used by the surrounding MoE kernels); rows are written at
    sorted_token_ids, and the A row of each slot is sorted_id // top_k.
    global_scale is the per-expert fp32 tensor ([E] for w2, [E,2] for w13 =>
    pass num_shards accordingly). moe_block_size is the route-block width that
    moe_align_block_size was called with (8 halves the wasted mma rows).
    """
    torch.ops._C.sm70_nvfp4_qpn2_moe_gemm(
        out, input, codes, scales, global_scale, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, int(top_k), bool(mul_topk_weights),
        int(num_shards), int(moe_block_size), bool(fuse_act))


# --- Round-3 TM-MoE shadow gate (R3 spec §2.3; R2-T1b house pattern) -------
def qpn2_moe_enabled() -> bool:
    """Run the MoE NVFP4 experts through the route-tiled QPN2 kernel.
    VLLM_SM70_MOE_QPN2 selects the implementation (same function, two
    kernels): 1 (default) = QPN2 route-tiled, 0 = the Marlin MoE grouped
    kernel. Measured on Qwen3.8-Flash-Next TP4xPP1: 67.5 vs 50.7 tok/s
    decode. Inert off exact-SM70 hardware (is_enabled fails closed).

    This is a *weight layout* switch, not a per-call one: when it is on the
    layer stores QPN2-packed expert weights and no Marlin copy is kept (both
    are 4-bit + group-scale, so it costs nothing, but keeping both would).
    """
    if envs.VLLM_SM70_MOE_QPN2 <= 0 or not is_enabled():
        return False
    try:
        return hasattr(torch.ops._C, "sm70_nvfp4_qpn2_moe_gemm")
    except Exception:
        return False


try:
    from torch.library import register_fake as _register_fake
except ImportError:
    from torch.library import impl_abstract as _register_fake


@_register_fake("_C::sm70_nvfp4_dequant")
def _sm70_nvfp4_dequant_fake(weight, weight_scale, global_scale):
    n = weight.shape[0]
    k = weight.shape[1] * 2
    return torch.empty(n, k, dtype=torch.float16, device=weight.device)


@_register_fake("_C::sm70_nvfp4_dequant_from_qpn2")
def _sm70_nvfp4_dequant_from_qpn2_fake(codes, scales, global_scale):
    n = codes.shape[0]
    k = codes.shape[1] * 2
    return torch.empty(n, k, dtype=torch.float16, device=codes.device)


@_register_fake("_C::sm70_nvfp4_qpn2_prepare")
def _sm70_nvfp4_qpn2_prepare_fake(weight, weight_scale):
    n = weight.shape[0]
    k = weight.shape[1] * 2
    dev = weight.device
    codes = torch.empty(n, k // 2, dtype=torch.uint8, device=dev)
    scales = torch.empty(n, k // 16, dtype=torch.uint8, device=dev)
    return codes, scales


@_register_fake("_C::sm70_nvfp4_qpn2_gemm")
def _sm70_nvfp4_qpn2_gemm_fake(out, input, codes, scales, global_scale,
                              split_k, nacc):
    return out


@_register_fake("_C::sm70_nvfp4_gemm_dispatch")
def _sm70_nvfp4_gemm_dispatch_fake(out, input, qpn2_codes, qpn2_scales,
                                   qpn2_global_scale, split_k, nacc):
    return None


@_register_fake("_C::sm70_nvfp4_qpn2_prepare_moe")
def _sm70_nvfp4_qpn2_prepare_moe_fake(weight, weight_scale):
    dev = weight.device
    codes = torch.empty_like(weight, dtype=torch.uint8, device=dev)
    scales = torch.empty_like(weight_scale, dtype=torch.uint8, device=dev)
    return codes, scales


@_register_fake("_C::sm70_nvfp4_qpn2_moe_gemm")
def _sm70_nvfp4_qpn2_moe_gemm_fake(out, input, codes, scales, global_scale,
                                   sorted_token_ids, expert_ids,
                                   num_tokens_past_padded, topk_weights,
                                   top_k, mul_topk_weights, num_shards,
                                   moe_block_size):
    return None


