# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal SM70 DFlash2 config adapter.

1cat's DFlash2 speculator reads a policy dict for optional SM70 extras
(sharded context fc, context kv graph, ...).  This tree does not ship those
extras, so the policy is always empty and every optional feature is off: the
speculator then runs its plain block-drafting path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DFlashLookupPolicy:
    """Inert lookup policy (successor table disabled)."""

    nstrong: int | None = None

    def resolve(self) -> None:
        if self.nstrong is None:
            self.nstrong = 0


@dataclass
class Sm70DFlash2Config:
    """Inert SM70 DFlash2 policy: every optional extra stays disabled.

    ``resolved`` stays False so ``sm70_dflash2_enabled`` falls through to this
    module's always-False implementation instead of reading policy fields.
    ``resolve`` only fills the two proposal filters the speculator asserts on,
    with the neutral values (temperature scale 1.0, top-p 1.0).
    """

    resolved: bool = False
    # The DFlash2 checkpoints declare bfloat16 weights, but SM70 has no native
    # BF16 arithmetic, so the draft runs in FP16. Without the range-preserving
    # path the MLP overflows FP16 and poisons every downstream layer with NaN.
    bf16_emulation: bool = True
    proposal_temperature_scale: float | None = None
    proposal_top_p: float | None = None
    lookup: DFlashLookupPolicy = field(default_factory=DFlashLookupPolicy)

    def resolve(self, *, qualified: bool = False) -> None:
        if self.proposal_temperature_scale is None:
            self.proposal_temperature_scale = 1.0
        if self.proposal_top_p is None:
            self.proposal_top_p = 1.0
        self.lookup.resolve()


def capture_sm70_dflash2_config(vllm_config: Any = None) -> dict[str, Any] | None:
    if vllm_config is None:
        return None
    spec = getattr(vllm_config, "speculative_config", None)
    extra = getattr(spec, "dflash2", None) if spec is not None else None
    return dict(extra) if isinstance(extra, dict) else {}


def sm70_dflash2_enabled(feature: str, config: Any) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(config.get(feature, False))

def dflash2_bf16_emulation(policy: "Sm70DFlash2Config | None") -> bool:
    """Whether the range-preserving FP16 path for BF16 drafts is enabled.

    Default on: the DFlash2 drafter is BF16-trained and SM70 lacks BF16
    arithmetic, so plain FP16 transport overflows.
    """
    if policy is not None:
        return bool(getattr(policy, "bf16_emulation", True))
    return True
