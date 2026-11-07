# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inert stand-in for 1cat's diagnostics plumbing.

The DFlash2 speculator only uses these for optional debug dumps; returning an
object that tolerates any attribute/call keeps the production path a no-op.
"""
from __future__ import annotations


class _Inert:
    def __getattr__(self, name: str) -> "_Inert":
        return _Inert()

    def __call__(self, *args, **kwargs) -> "_Inert":
        return _Inert()

    def __bool__(self) -> bool:
        return False


def bind_diagnostics(*args, **kwargs) -> None:
    return None


def diagnostics_for(*args, **kwargs) -> None:
    return None


def diagnostic_channel(*args, **kwargs) -> _Inert | None:
    return None


def write_payload(*args, **kwargs) -> None:
    return None
