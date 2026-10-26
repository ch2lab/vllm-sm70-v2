# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sm70_nvfp4_ops import gemm_dispatch, is_enabled, qpn2_prepare

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class Sm70NvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM on SM70: single QPN2 4-bit layout (1x storage).

    Decode (m <= 32): QPN2 high-occupancy 4-bit GEMM (reads the QPN2 layout
    directly, no dequant).
    Prefill (m > 32): dequant the QPN2 layout to FP16 (transient scratch) +
    cuBLAS FP16 GEMM with fp32 accumulation.

    Storage: 1x (single QPN2 4-bit layout). The Marlin layout is dropped;
    on V100 the dequant+cuBLAS path beats the Marlin 4-bit GEMM (1.25-1.39x).
    """

    @classmethod
    def is_supported(cls, compute_capability=None):
        if compute_capability is None:
            _cc = current_platform.get_device_capability()
            compute_capability = _cc.to_int() if _cc is not None else None
        if compute_capability != 70:
            return False, "SM70 NVFP4 requires SM70 (7.0)"
        if not is_enabled():
            return False, "SM70 NVFP4 op unavailable in the loaded extension"
        return True, None

    @classmethod
    def can_implement(cls, config):
        return True, None

    def process_weights_after_loading(self, layer):
        for attr in ("weight", "weight_scale", "weight_global_scale"):
            if not hasattr(layer, attr):
                raise AttributeError(
                    f"SM70 NVFP4 kernel requires layer.{attr}")
        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        # QPN2 requires n % 32 == 0. A layer whose output dim is not a
        # multiple of 32 is zero-padded: the extra rows dequantize to 0 and
        # are sliced off in apply_weights.
        n_pad = (n + 31) // 32 * 32
        weight, scales = layer.weight, layer.weight_scale
        if n_pad != n:
            pad = n_pad - n
            weight = torch.nn.functional.pad(weight, (0, 0, 0, pad), value=0)
            scales = torch.nn.functional.pad(scales, (0, 0, 0, pad), value=0)
        qpn2_codes, qpn2_scales = qpn2_prepare(weight, scales)
        layer._sm70_qpn2_codes = qpn2_codes
        layer._sm70_qpn2_scales = qpn2_scales
        gs = layer.weight_global_scale
        if torch.is_tensor(gs):
            gs = float(gs.item())
        layer._sm70_qpn2_global_scale = gs
        if n_pad != n:
            layer._sm70_nvfp4_padded_n = n_pad
        del layer.weight
        logger.info_once("SM70: n=%d (gemm_n=%d) QPN2 1x.", n, n_pad)

    def apply_weights(self, layer, x, bias=None):
        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        # The GEMM accumulates in fp32 (CUBLAS_COMPUTE_32F) and writes fp16.
        # The model runs fp16, so x and c are both fp16 and no dtype round-trip
        # is needed -- an fp16->fp32->fp16 cast was a lossless no-op that
        # cost two full-size copies per GEMM.
        a = x.reshape(-1, k).to(torch.float16)
        m = a.shape[0]
        n_gemm = getattr(layer, "_sm70_nvfp4_padded_n", n)
        c = torch.empty(m, n_gemm, dtype=torch.float16, device=a.device)
        # Per-shape QPN2 template (microbench 2026-09-21, m=1): small-N GEMMs
        # (few CTAs) gain from SplitK=16 (16-warp block), nacc=2 pipelines
        # the mma chains; large-N GEMMs are best at 8/1. Grid-z K-split
        # (VLLM_SM70_QPN2_KSPLIT) measured regressive on all shapes — leave
        # off.
        if n_gemm <= 5120 and (k // 16) % 16 == 0:
            split_k, nacc = 16, (2 if k <= 5120 else 1)
        else:
            split_k, nacc = 8, 1
        gemm_dispatch(
            c, a,
            layer._sm70_qpn2_codes, layer._sm70_qpn2_scales,
            layer._sm70_qpn2_global_scale,
            split_k, nacc)
        if n_gemm != n:
            c = c[:, :n]
        if bias is not None:
            c = c + bias.to(torch.float16)
        return c.to(x.dtype).reshape(*x.shape[:-1], n)
