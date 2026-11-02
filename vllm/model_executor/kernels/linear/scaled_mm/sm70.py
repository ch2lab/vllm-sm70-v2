# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)

def _qpn_available() -> bool:
    return (
        envs.VLLM_SM70_FP8_QPN
        and current_platform.is_cuda()
        and current_platform.get_device_capability().to_int() == 70
        and hasattr(torch.ops._C, "sm70_fp8_qpn_pack")
    )


def _weight_as_input_major(
    w: torch.Tensor, layer: torch.nn.Module
) -> torch.Tensor:
    """Return the FP8 weight as a row-major [input, output] tensor.

    Which axis is the contraction axis is not uniform across FP8 methods (and
    the two are frequently the same length under TP), so the parameter's own
    axis metadata decides; it is kept in step with the data by the in-place
    transposes the quant methods apply. The layer's sizes only serve as a
    fallback for a plain parameter. Ambiguity raises: silently reading a
    transposed weight would corrupt the model, not fail.
    """
    input_dim = getattr(w, "input_dim", None)
    if input_dim == 0:
        return w.contiguous()
    if input_dim == 1:
        return w.t().contiguous()
    k = getattr(layer, "input_size_per_partition", None)
    n = getattr(layer, "output_size_per_partition", None)
    if k is None:
        k = layer.input_size
    if n is None:
        n = layer.output_size
    if k == n:
        raise ValueError(
            f"SM70 FP8 QPN: a square {tuple(w.shape)} FP8 weight without "
            f"input_dim metadata cannot be oriented safely"
        )
    if tuple(w.shape) == (k, n):
        return w.contiguous()
    if tuple(w.shape) == (n, k):
        return w.t().contiguous()
    raise ValueError(
        f"SM70 FP8 QPN: cannot resolve the axes of a {tuple(w.shape)} weight "
        f"(input_size_per_partition={k}, output_size_per_partition={n}, "
        f"no input_dim metadata)"
    )


class Sm70Fp8DequantLinearKernel(FP8ScaledMMLinearKernel):
    """FP8-for-16-bit linears on exact SM70 (V100/Volta).

    Volta has no FP8 hardware, so every native FP8 GEMM backend (CUTLASS,
    FlashInfer, Marlin, torch._scaled_mm) is unusable there. Two paths replace
    them:

    * **QPN fragment kernel** (default): the weight stays in its 1-byte FP8
      encoding, pre-packed at load into the lane-major operand layout the NVFP4
      QPN2 kernel uses, and each byte is expanded in registers straight into one
      `mma.m8n8k4` B-fragment. Decode therefore streams half the bytes of any
      16-bit GEMM, and there is no FP16 weight copy to keep resident. E4M3 ->
      FP16 is exact, so the only arithmetic difference from the FP16 path is the
      FP32 accumulation order.
    * **Dequantise-to-FP16 + cuBLAS**: used for m > 16 and for shapes the
      fragment kernel cannot take. The FP16 operand is rebuilt from the packed
      bytes on demand (`sm70_fp8_qpn_dequant`, bit-identical to the old
      load-time dequantise), so this path costs bandwidth on large batches
      instead of memory on every batch.

    A scalar register-dequant GEMV is also implemented and tested as the
    `sm70_fp8_gemv` op, but is deliberately not wired in: on V100 it tops out at
    ~260 GB/s because every byte pays for an FP32 multiply, against ~490 GB/s
    for the fragment feed above. `VLLM_SM70_FP8_QPN` selects the feed
    (True = QPN fragment feed, False = FP16 path for every batch size).
    """

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence
    ) -> None:
        super().__init__(c, layer_param_names)
        self._weight_deq: torch.Tensor | None = None
        self._qpn: torch.Tensor | None = None
        self._qpn_axes: tuple[int, int] | None = None
        # per-channel weight scale: the scalar the fragment feed takes is the
        # MAX of the row scales (keeps every intermediate in fp16 range), and
        # the per-row residual is multiplied back after the GEMM.
        self._qpn_row_scale: torch.Tensor | None = None
        self._qpn_wscale: torch.Tensor | None = None

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        if compute_capability is None:
            cap = current_platform.get_device_capability()
            compute_capability = cap.to_int() if cap is not None else None
        if compute_capability == 70:
            return True, None
        return False, "requires exact SM70 (V100/Volta)."

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
            and c.weight_quant_key.scale.group_shape.is_per_tensor()
        ):
            return True, None
        # w8a16: 16-bit runtime activations (the dynamic key never quantizes
        # the input at runtime) may carry per-channel weight scales; the
        # scale is baked into the FP16 weight copy at load time.
        if (
            c.input_dtype == torch.float16
            and c.activation_quant_key != kFp8StaticTensorSym
        ):
            return True, None
        return False, "requires per-tensor scales, or FP16 runtime inputs."

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def get_output_padding(self) -> int | None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, w_s = layer.weight, layer.weight_scale
        # The QPN feed passes the weight scale straight to
        # sm70_fp8_qpn_dispatch, which takes a scalar: per-channel scales
        # must go through the FP16 dequantise path instead (where the
        # channelwise multiply happens at load).
        qpn_eligible = self.config.weight_quant_key.scale.group_shape.is_per_tensor()
        if not qpn_eligible and _qpn_available() and self.config.input_dtype == torch.float16:
            ws = w_s.detach().float().flatten()
            out_f = (getattr(layer, "output_size_per_partition", None)
                     or getattr(layer, "num_embeddings", None)
                     or getattr(layer, "out_features", None))
            in_f = (getattr(layer, "input_size_per_partition", None)
                    or getattr(layer, "embedding_dim", None)
                    or getattr(layer, "in_features", None))
            # Orientation: this quant method stores the weight input-major
            # ([in, out]) -- `_weight_deq = w * s` is fed straight to
            # `a @ B` below -- so that branch must be tested first. Under TP a
            # RowParallelLinear shard is square ([in/tp, in/tp]) and both
            # predicates hold; testing the output-major one first would
            # transpose the weight and silently corrupt every GDN/attention
            # out_proj (TP1 hides it because its shard is never square).
            src = None
            if out_f is not None and in_f is not None:
                if tuple(w.shape) == (int(in_f), int(out_f)):
                    src = w
                elif tuple(w.shape) == (int(out_f), int(in_f)):
                    src = w.t()
            if (
                src is not None
                and ws.numel() == int(out_f)
                and int(src.shape[1]) == int(out_f)
            ):
                src = src.contiguous()
                k, n = int(src.shape[0]), int(src.shape[1])
                if n % 32 == 0 and k % 16 == 0:
                    self._qpn = torch.ops._C.sm70_fp8_qpn_pack(src.view(torch.uint8))
                    self._qpn_axes = (k, n)
                    wmax = ws.max().reshape(1)
                    self._qpn_wscale = wmax
                    self._qpn_row_scale = (ws / wmax).view(1, n)
                    if hasattr(layer, "weight"):
                        del layer.weight
                    return
            logger.warning(
                "SM70 FP8 QPN per-channel skipped for %s: w=%s out=%s in=%s scales=%d",
                layer, tuple(w.shape), out_f, in_f, int(ws.numel()))
        if (
            _qpn_available()
            and self.config.input_dtype == torch.float16
            and qpn_eligible
        ):
            try:
                src = _weight_as_input_major(w, layer)
            except ValueError as e:
                # An unresolvable axis order is a reason to take the slower
                # path, not to refuse to serve: the FP16 dequantise path below
                # is orientation-agnostic. Loud, because it means a quant
                # method showed up that this kernel has not been taught about.
                logger.warning("SM70 FP8 QPN disabled for %s: %s", layer, e)
            else:
                k, n = int(src.shape[0]), int(src.shape[1])
                if n % 32 == 0 and k % 16 == 0:
                    self._qpn = torch.ops._C.sm70_fp8_qpn_pack(
                        src.view(torch.uint8)
                    )
                    self._qpn_axes = (k, n)
                    # Nothing but the packed bytes may stay resident: the
                    # 16-bit copy this path replaces is ~0.64 GB per rank of
                    # dead weight for Ornith's projections.
                    if hasattr(layer, "weight"):
                        del layer.weight
                    return
                logger.warning(
                    "SM70 FP8 QPN skipped for %s: needs output %% 32 == 0 and "
                    "input %% 16 == 0, got k=%d n=%d",
                    layer,
                    k,
                    n,
                )
        w_f = w.float()
        s = w_s.float()
        # Per-channel (per-output-channel) weight scale: align to the OUTPUT
        # axis. The dequant weight is [input, output] (input-major) here, so the
        # output is dim1. When the scale size matches only one dim the old
        # numel heuristic was fine, but on a SQUARE weight (e.g. GDN out_proj
        # [2048, 2048] at TP2) it matched dim0 and silently scaled the wrong
        # axis. Disambiguate with the layer's output_size_per_partition,
        # preferring the output dim (dim1) on a square weight.
        _outp = getattr(layer, "output_size_per_partition", None)
        if _outp is not None and s.numel() == _outp:
            if w_f.shape[1] == _outp:
                s = s.reshape(1, -1)
            else:
                s = s.reshape(-1, 1)
        elif s.numel() == w_f.shape[0]:
            s = s.reshape(-1, 1)
        elif s.numel() == w_f.shape[1]:
            s = s.reshape(1, -1)
        self._weight_deq = (w_f * s).to(self.config.input_dtype)
        if os.environ.get("VLLM_SM70_FP8_LOADDBG") == "1":
            import sys as _fsys
            import torch.distributed as _fdist
            _frank = (
                _fdist.get_rank() if _fdist.is_initialized() else 0
            )
            _w = w.detach()
            _deq = self._weight_deq
            print(
                f"[F8LOAD] r{_frank} {getattr(layer, 'prefix', '?')[-46:]} "
                f"wshp={tuple(_w.shape)} wsum={_w.float().sum().item():.6g} "
                f"wshp_raw={tuple(w_s.shape)} "
                f"ssum={w_s.float().sum().item():.6g} "
                f"srow0={w_s.float().flatten()[0].item():.6g} "
                f"srowL={w_s.float().flatten()[-1].item():.6g} "
                f"deqsum={_deq.sum().item():.6g} "
                f"deqmax={_deq.abs().max().item():.6g} "
                f"deqshp={tuple(_deq.shape)} "
                f"d0half={_deq[: _deq.shape[0] // 2].sum().item():.6g} "
                f"d1half={_deq[:, : _deq.shape[1] // 2].sum().item():.6g}",
                file=_fsys.stderr,
                flush=True,
            )
        if hasattr(layer, "weight"):
            del layer.weight

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = self.layer_param_names
        if self._qpn is not None:
            view = self._qpn.view(self._qpn_axes)
            if self._qpn_row_scale is not None:
                return (view, self._qpn_wscale,
                        getattr(layer, x_s, None), getattr(layer, x_s_ub, None))
            return (
                view,
                getattr(layer, w_s),
                getattr(layer, x_s, None),
                getattr(layer, x_s_ub, None),
            )
        if self._weight_deq is None:
            self.process_weights_after_loading(layer)
        return (
            self._weight_deq,
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # A arrives FP8 (or pre-quantized); E4M3 -> FP16 is exact, so the
        # compute dtype is FP16 either way.
        if A.dtype == self.fp8_dtype:
            if (
                out_dtype == torch.float16
                and A.is_contiguous()
                and A.numel() % 8 == 0
                and As.numel() == 1
            ):
                # One kernel instead of the cast + multiply-and-cast pair that
                # the eager expression below compiles to, bit-for-bit the same
                # values.
                a = torch.ops._C.sm70_fp8_dequant_act(A, As)
            else:
                a = (A.float() * As).to(out_dtype)
        else:
            a = A
        if self._qpn is not None:
            # One op, and the batch-size decision lives inside it: this runs
            # under torch.compile, where `a.shape[0]` is a symbolic integer, and
            # a Python `if` on it bakes whichever branch the trace happened to
            # take into the artifact used by every other batch size.
            out = torch.empty(
                a.shape[0],
                self._qpn_axes[1],
                dtype=torch.float16,
                device=a.device,
            )
            torch.ops._C.sm70_fp8_qpn_dispatch(out, a, self._qpn, Bs, 16)
            if self._qpn_row_scale is not None:
                out = (out.float() * self._qpn_row_scale).to(out.dtype)
        else:
            out = a @ B
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


try:
    from torch.library import register_fake as _register_fake
except ImportError:  # torch < 2.4
    from torch.library import impl_abstract as _register_fake


@_register_fake("_C::sm70_fp8_qpn_pack")
def _sm70_fp8_qpn_pack_fake(weight):
    return torch.empty(weight.shape, dtype=torch.uint8, device=weight.device)


@_register_fake("_C::sm70_fp8_qpn_dequant")
def _sm70_fp8_qpn_dequant_fake(packed, scale, k, n):
    return torch.empty(k, n, dtype=torch.float16, device=packed.device)


@_register_fake("_C::sm70_fp8_qpn_gemm")
def _sm70_fp8_qpn_gemm_fake(out, input, packed, scale, split_k):
    return None


@_register_fake("_C::sm70_fp8_qpn_dispatch")
def _sm70_fp8_qpn_dispatch_fake(out, input, packed, scale, split_k):
    return None


@_register_fake("_C::sm70_fp8_dequant_act")
def _sm70_fp8_dequant_act_fake(x_fp8, scale):
    return torch.empty(x_fp8.shape, dtype=torch.float16, device=x_fp8.device)
