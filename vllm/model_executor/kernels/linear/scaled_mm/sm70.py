# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        return False, "requires per-tensor weight and activation scales."

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def get_output_padding(self) -> int | None:
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, w_s = layer.weight, layer.weight_scale
        if _qpn_available() and self.config.input_dtype == torch.float16:
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
        self._weight_deq = (w.float() * w_s).to(self.config.input_dtype)
        if hasattr(layer, "weight"):
            del layer.weight

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = self.layer_param_names
        if self._qpn is not None:
            view = self._qpn.view(self._qpn_axes)
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
