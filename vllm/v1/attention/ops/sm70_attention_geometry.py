# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project

"""Single source of truth for SM70 fast-kernel geometry admission.

Every hand-written SM70 attention kernel carries shape assumptions (head count,
KV-head ratio, page layout, stride).  A kernel that is enabled on an unverified
geometry does not crash -- it silently computes the wrong attention, which is
far worse.  So admission is explicit and evidence-based:

* the default is **deny** -- anything not listed here falls back to the generic
  Triton implementation, which is slow but defines correctness;
* a geometry is listed only after it passes the oracle parity gate
  (``sm70_tests/test_geometry_parity.py``: rel <= 5e-4 against the generic
  kernel, bit-repeatable, no out-of-shape writes) and the behavioural gates
  (``quality_secret.py`` and friends).

Evidence (2026-11-03, ``test_geometry_parity.py``, 2048-token cache):

  geometry   prefill/twopass  prefill/cuda  decode/xqa
  (2, 16)    3.660e-04       6.245e-05     5.432e-05
  (1, 8)     3.661e-04       6.202e-05     4.111e-05
  (1, 4)     3.664e-04       6.179e-05     5.813e-05

Do not add a geometry "because it runs".  See ``sm70/27_phase_charter.md``.
"""

from __future__ import annotations

VERIFIED_GEOMETRIES: frozenset[tuple[int, int]] = frozenset(
    {
        (2, 16),  # TP1: 2 KV heads, 16 qo heads (ratio 8)
        (1, 8),  # TP2: 1 KV head, 8 qo heads (ratio 8)
        (1, 4),  # TP4: 1 KV head, 4 qo heads (ratio 4)
    }
)


def supported(H: int, HQ: int, D: int = 256) -> bool:
    """Whether the fast prefill/decode kernels may run this geometry."""
    return D == 256 and (H, HQ) in VERIFIED_GEOMETRIES


def supported_decode(H: int, HQ: int, D: int = 256) -> bool:
    """Decode used to require HQ == H * 8 (compile-time RATIO); since the
    ratio is now derived per build, it has no extra constraint."""
    return supported(H, HQ, D)
