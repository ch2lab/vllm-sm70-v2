# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide HC sequence-parallel phase flag.

``--enable-hc-sp`` is a phase-preference knob on SM70: the per-layer
``sp_all_gather`` costs a fixed ~0.3 ms x 48 layers per *decode* step (which a
1-token step cannot amortize) while long prefill batches gain ~24%. This flag
lets the engine run SP only for prefill and the replicated path for decode.

Set by the model runner once per forward (``has_prefill``); the CUDA-graph
dispatch bakes the branch at capture time (decode graphs are captured from
uniform decode dummy batches, prefill graphs from prefill batches), so replay
needs no Python.

``VLLM_SM70_HC_SP_PHASE=1`` enables the behaviour. With it unset the phase is
always "active" and the engine behaves exactly as before.
"""

from __future__ import annotations

import os

# NOTE (2026-10-23): this is deliberately NOT env-gated. The engine/worker
# processes do not inherit most VLLM_SM70_* variables (the worker's environment
# is curated; only ~11 VLLM_ vars survive), so an env-gated valve is silently
# dead in the process that runs the model. The boot switch is --enable-hc-sp:
# with it set, the phase selector is active; remove the flag to get the old
# always-shard behaviour back.
_ENABLED = True

# True = shard tokens (sequence parallel), False = replicate tokens.
_active = True


def enabled() -> bool:
    return _ENABLED


def set_active(active: bool) -> None:
    global _active
    _active = bool(active)


def active() -> bool:
    return _active


def sp_now(boot_flag: bool) -> bool:
    """Resolve the per-forward SP decision from the boot flag and the phase."""
    return boot_flag and (not _ENABLED or _active)


_logged: set[bool] = set()


def should_log(active: bool) -> bool:
    """First time each phase is entered (debug builds only)."""
    if active in _logged:
        return False
    _logged.add(active)
    return True
