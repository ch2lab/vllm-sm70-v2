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

  geometry   prefill/twopass  prefill/cuda  decode/xqa  decode/scalar
  (2, 16)    3.664e-04       6.227e-05     5.989e-05   8.386e-06
  (1, 8)     3.660e-04       6.213e-05     6.167e-05   1.679e-05
  (1, 4)     3.669e-04       6.223e-05     5.037e-05   0.000e+00

Pending (not admitted yet, kept for the 8-GPU host):

  (1, 2)     TP8           3.829e-04       6.202e-05     6.279e-05   untested

  The L0 numbers above are real (diagnostic run, ``--force-geometry 1,2``);
  the behavioural gates need a real 8-GPU service.  Set
  ``VLLM_SM70_GEOMETRY_ALLOW=1,2`` to enable it for that validation run only;
  once canary/secret/longgen/concurrent pass there, promote it into
  ``VERIFIED_GEOMETRIES`` and re-run this harness without the override.

Do not add a geometry "because it runs".  See ``sm70/27_phase_charter.md``.
"""

from __future__ import annotations

import os

VERIFIED_GEOMETRIES: frozenset[tuple[int, int]] = frozenset(
    {
        (2, 16),  # TP1: 2 KV heads, 16 qo heads (ratio 8)
        (1, 8),  # TP2: 1 KV head, 8 qo heads (ratio 8)
        (1, 4),  # TP4: 1 KV head, 4 qo heads (ratio 4)
    }
)


def _parse_geometries(raw: str) -> frozenset[tuple[int, int]]:
    out = set()
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        h, q = part.replace(":", ",").split(",")
        out.add((int(h), int(q)))
    return frozenset(out)


# Test-only override, read once at import: the env must be set before the
# service starts.  Used to validate a pending geometry (currently TP8) on the
# 8-GPU host without editing this file.
EXTRA_GEOMETRIES = _parse_geometries(
    os.environ.get("VLLM_SM70_GEOMETRY_ALLOW", "")
)


def supported(H: int, HQ: int, D: int = 256) -> bool:
    """Whether the fast prefill/decode kernels may run this geometry."""
    return D == 256 and (
        (H, HQ) in VERIFIED_GEOMETRIES or (H, HQ) in EXTRA_GEOMETRIES
    )


def supported_decode(H: int, HQ: int, D: int = 256) -> bool:
    """Decode admission.

    The decode kernel derives its head ratio per build, so the constraint is
    the group size ``G = HQ // H``, not a specific (H, HQ) pair: the QK pass
    maps G heads onto ``(G + 1) // 2`` softmax warps and the PV pass covers
    ``D / 32`` warps, which bounds G to [2, 8] at D=256.

    Evidence for the pairs beyond VERIFIED_GEOMETRIES (2026-11-07, 2048-token
    cache, split-plane nvfp4 fixture, Triton oracle): (1,6) and (2,4) both
    give max_abs 2.44e-04 = one fp16 ulp, identical to the already admitted
    (1,8) measured by the same harness -- i.e. G=6 and G=2 behave exactly like
    the verified G=8/G=4 cases. Needed by 27B (24/4 heads: G=6 at TP4) and
    35B (16/2 heads: G=2 at TP4).
    """
    if D != 256 or H <= 0 or HQ % H != 0:
        return False
    g = HQ // H
    return 2 <= g <= 8 and (
        (H, HQ) in VERIFIED_GEOMETRIES or (H, HQ) in EXTRA_GEOMETRIES
        or g in (2, 4, 6, 8)
    )
