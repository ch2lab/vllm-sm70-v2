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

    def __int__(self) -> int:
        return 0

    def value(self, name: str) -> int:
        return 0


def bind_diagnostics(*args, **kwargs) -> "_Inert":
    return _Inert()


def diagnostics_for(*args, **kwargs) -> "_Inert":
    # Callers do `diagnostics_for(cfg).sampling` and friends; the inert object
    # keeps those chains valid while `__bool__` stays False so they still read
    # as "diagnostics disabled".
    return _Inert()


def diagnostic_channel(*args, **kwargs) -> _Inert:
    return _Inert()


def write_payload(*args, **kwargs) -> None:
    return None
