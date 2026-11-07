# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal SM70 DFlash2 config adapter.

1cat's DFlash2 speculator reads a policy dict for optional SM70 extras
(sharded context fc, context kv graph, ...).  This tree does not ship those
extras, so the policy is always empty and every optional feature is off: the
speculator then runs its plain block-drafting path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Sm70DFlash2Config:
    """Inert SM70 DFlash2 policy: every optional extra stays disabled.

    ``resolved=False`` makes the speculator's ``sm70_dflash2_enabled`` lookups
    fall through to this module's own (always False) implementation instead of
    reading per-field policy attributes.
    """

    resolved: bool = False


def capture_sm70_dflash2_config(vllm_config: Any) -> dict[str, Any]:
    spec = getattr(vllm_config, "speculative_config", None)
    extra = getattr(spec, "dflash2", None) if spec is not None else None
    return dict(extra) if isinstance(extra, dict) else {}


def sm70_dflash2_enabled(feature: str, config: Any) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(config.get(feature, False))
