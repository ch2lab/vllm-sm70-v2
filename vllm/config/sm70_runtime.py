# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert stand-in for 1cat's runtime trace plumbing.

The ported DFlash draft model asks the runtime trace config whether optional
context-KV tracing is on (``capture_runtime_trace().dflash.value("context_kv")``
and similar).  This tree has no such tracing, so every answer is 0 (off).
"""
from __future__ import annotations


class _TraceChannel:
    def value(self, name: str) -> int:
        # Opt-in tracing: QUASAR_TRACE_PROPOSAL_STAGES=1 enables the ported
        # DFlash2 token diagnostic without touching the speculator.
        import os

        return int(os.environ.get("QUASAR_TRACE_" + name.upper(), "0"))

    def __getattr__(self, name: str) -> "_TraceChannel":
        return _TraceChannel()


class RuntimeTraceConfig:
    dflash = _TraceChannel()

    def __getattr__(self, name: str) -> _TraceChannel:
        return _TraceChannel()


def capture_runtime_trace() -> RuntimeTraceConfig:
    return RuntimeTraceConfig()
