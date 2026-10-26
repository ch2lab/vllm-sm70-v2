# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot-time audit of the SM70 valve environment in the process that runs the model.

Rationale: the serve script exports the tuned valves, but the model executes in
worker processes. A silent mismatch (valve missing -> library default) is
invisible in the logs and has cost this project a documented 13% gap incident.
This audits once per process, logs the full set actually seen, and fails the
boot when a required valve is missing instead of degrading silently.

Only the IPC serve is audited: the preload daemon loads from disk and is
expected to run with a different (minimal) environment.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# Valves the production serve script sets on purpose. Keep this list in sync
# with run_sm70_serve_ipc.sh; a deliberate removal must also remove it here.
REQUIRED: tuple[str, ...] = (
    "VLLM_SM70_ALLOW_DUP_LAYER",
    "VLLM_SM70_AR_ADAPTIVE",
    "VLLM_SM70_AR_HIER",
    "VLLM_SM70_AR_WIRE",
    "VLLM_SM70_GDN_CHUNKED",
    "VLLM_SM70_MOE_QPN2_NS",
    "VLLM_SM70_MOE_QPN2_WIDE",
    "VLLM_SM70_PLE_SYNC",
    "VLLM_SM70_QSA_DECODE_FUSE",
    "VLLM_SM70_QSA_DRAFT_META_UPDATE",
    "VLLM_SM70_QSA_PAGE4_SORT",
    "VLLM_SM70_QSA_PLAN_PACK",
    "VLLM_SM70_QSA_POS_FUSE",
    "VLLM_SM70_QSA_STAGES",
    "VLLM_SM70_QSA_VERIFY_PAGE_CAP",
    "VLLM_SM70_QWEN38_FP16_GEMV",
    "VLLM_SM70_QWEN38_FUSED_HC_DOWN_SILU",
    "VLLM_SM70_QWEN38_FUSED_HC_UP_GATE",
    "VLLM_SM70_TRITON_TOPK",
)

_audited = False


def audit_once(*, strict: bool = True) -> None:
    """Log the SM70 valve set once per process and fail on missing valves."""
    global _audited
    if _audited:
        return
    _audited = True

    seen = {k: v for k, v in os.environ.items() if k.startswith("VLLM_SM70")}
    missing = [k for k in REQUIRED if k not in seen]
    logger.info(
        "[SM70-ENV] pid=%d sees %d SM70 vars: %s",
        os.getpid(),
        len(seen),
        ", ".join(f"{k}={seen[k]}" for k in sorted(seen)),
    )
    if missing:
        msg = (
            "[SM70-ENV] MISSING valves in the process that runs the model "
            f"(defaults will be used instead): {missing}"
        )
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)
