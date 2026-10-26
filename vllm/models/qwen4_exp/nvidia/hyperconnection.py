# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .opattr_hc import hc_flush, hc_mark
from .ops.cute_dsl.hc_down_silu import MAX_FUSED_M, hc_down_silu
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .sm70_fp16_hc import (
    maybe_apply_qwen38_sm70_fp16_fused_hc,
    maybe_apply_qwen38_sm70_fp16_hc_down_silu,
    maybe_apply_qwen38_sm70_fp16_hc_up_gate,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject).
    Eligible down+inject projections use the fused SiLU GEMM; other projections
    use their Linear module's GEMM dispatch.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_lowrank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.hc_lowrank + self.hc_count)) % 16 if use_combine else 0
        # #197: TP-shard the low-rank down+inject GEMM (RowParallel = per-rank
        # K slice + small all-reduce) — 8x down-GEMM FLOPs cut at TP8. The
        # checkpoint already stores down/inject separately, so the sharded
        # leaves load 1:1 (K-dim narrow, no packing). Default off = the merged
        # path, byte-identical to today.
        self._hc_sharded = use_combine and envs.VLLM_SM70_HC_TP
        if use_combine and not self._hc_sharded:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.hc_lowrank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
            weight = self.input_mix_weight_down_block_inject.weight
            self._use_hc_down_silu = (
                weight.shape[1] % 8 == 0
                and weight.dtype == torch.bfloat16
                and current_platform.has_device_capability(90)
            )
        elif self._hc_sharded:
            # #197 sharded leaves: leaf names match the separate checkpoint
            # tensors exactly, so the loader loads them 1:1 with the
            # RowParallel K-dim narrow (no stacked/padded merged module).
            self.input_mix_weight_down = RowParallelLinear(
                self.hyper_hidden_size,
                self.hc_lowrank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
                input_is_parallel=False,
            )
            self.block_inject_weight = RowParallelLinear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "block_inject_weight"),
                return_bias=False,
                input_is_parallel=False,
            )
            self._use_hc_down_silu = False
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.hc_lowrank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.hc_lowrank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def _down_and_inject(
        self, xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Down projection + SiLU; also returns injection logits if combined."""
        if not self.use_combine:
            return hc_silu(self.input_mix_weight_down(xn), self.hc_count), None

        if self._hc_sharded:
            # #197: per-rank K-slice GEMMs; each module all-reduces its own
            # partial ([M, lora_rank] / [M, hc_count] fp16 — small AR).
            lora = self.input_mix_weight_down(xn)
            injection = self.block_inject_weight(xn)
            return hc_silu(lora, self.hc_count), injection

        use_fused = (
            self._use_hc_down_silu
            and not envs.VLLM_BATCH_INVARIANT
            and 1 <= xn.shape[0] <= MAX_FUSED_M
        )
        if use_fused:
            return hc_down_silu(
                xn,
                self.input_mix_weight_down_block_inject.weight,
                self.hc_lowrank,
                self.hc_count,
            )
        down_and_injection = self.input_mix_weight_down_block_inject(xn)
        split_sizes = [self.hc_lowrank, self.hc_count, self.pad_size]
        lowrank, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        return hc_silu(lowrank, self.hc_count), injection

    def _project(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """SM70 fp16 fused HC shortcut, falling back to the standard path."""
        if self.use_combine and not self._hc_sharded:
            # #197: the fused shortcut consumes the merged module's weight;
            # the sharded leaves have no merged module, so it stays off.
            fused_fp16 = maybe_apply_qwen38_sm70_fp16_fused_hc(
                self.input_mix_weight_down_block_inject,
                self.input_mix_weight_up,
                xn,
                getattr(self, "_sm70_qwen38_fp16_fused_hc", False),
            )
            if fused_fp16 is not None:
                hc_mark("hc:fused")
                return fused_fp16

        fused_down = None
        down_proj = getattr(self, "input_mix_weight_down_block_inject", None)
        if down_proj is None:
            down_proj = getattr(self, "input_mix_weight_down", None)
        if down_proj is not None and getattr(
            self, "_sm70_qwen38_fp16_down_silu", False
        ):
            fused_down = maybe_apply_qwen38_sm70_fp16_hc_down_silu(
                down_proj.weight, xn, True
            )
        if fused_down is None:
            lora, injection = self._down_and_inject(xn)
        else:
            lora, injection = fused_down
        hc_mark("hc:down")
        fused_up_gate = maybe_apply_qwen38_sm70_fp16_hc_up_gate(
            self.input_mix_weight_up.weight,
            xn,
            lora,
            getattr(self, "_sm70_qwen38_fp16_up_gate", False),
        )
        if fused_up_gate is not None:
            hc_mark("hc:up_gate")
            return fused_up_gate, injection
        gate = self.input_mix_weight_up(lora)  # [M, D]
        hc_mark("hc:up")
        out = hc_gate_mix(xn, gate, self.hc_count)
        hc_mark("hc:gate")
        return out, injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hc_mark("hc:begin")
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = self._project(xn)

        hc_flush("hc:mix")
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm. A missing injection applies the block output
        to every stream with unit weight.
        """
        hc_mark("hc:begin")
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = self._project(xn)

        hc_flush("hc:combine_and_mix")
        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
