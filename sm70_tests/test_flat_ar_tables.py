#!/usr/bin/env python3
"""Table-derivation check for the flat AR route (CPU only, no GPU work).

  /data/vllm-dev/bin/python sm70_tests/test_flat_ar_tables.py

Covers the topology cases the route has to handle:
  * TP8 on this box (all pairs report peer access, cross-NUMA pairs are SYS):
    the SYS peers must be routed as 2-hop over NVLink, never as direct links;
  * TP4 on 0-3 / 4-7 (fully NVLink connected): single hop, no relays;
  * a synthetic mixed box (NVLink + PCIe switch + SYS) where a PCIe-switch
    rank is the only usable relay;
  * a group where peer access is one-way for one pair: the pair is dropped.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/data/src/vllm-v2")

from vllm.distributed.device_communicators.sm70_flat_ar import (  # noqa: E402
    derive_tables,
)

OLD_N = {0: [1, 2, 3, 7], 1: [0, 3, 2, 6], 2: [0, 5, 1, 3], 3: [1, 4, 0, 2],
         4: [3, 6, 5, 7], 5: [2, 7, 4, 6], 6: [4, 7, 1, 5], 7: [5, 6, 0, 4]}
OLD_S = {0: [4, 5, 6], 1: [4, 5, 7], 2: [4, 6, 7], 3: [5, 6, 7],
         4: [0, 1, 2], 5: [0, 1, 3], 6: [0, 2, 3], 7: [1, 2, 3]}

TOP8 = [  # measured: `nvidia-smi topo -m` on this box
    ["X", "NV2", "NV2", "NV1", "SYS", "SYS", "SYS", "NV1"],
    ["NV2", "X", "NV1", "NV2", "SYS", "SYS", "NV1", "SYS"],
    ["NV2", "NV1", "X", "NV1", "SYS", "NV2", "SYS", "SYS"],
    ["NV1", "NV2", "NV1", "X", "NV2", "SYS", "SYS", "SYS"],
    ["SYS", "SYS", "SYS", "NV2", "X", "NV1", "NV2", "NV1"],
    ["SYS", "SYS", "NV2", "SYS", "NV1", "X", "NV1", "NV2"],
    ["SYS", "NV1", "SYS", "SYS", "NV2", "NV1", "X", "NV2"],
    ["NV1", "SYS", "SYS", "SYS", "NV1", "NV2", "NV2", "X"],
]


def check(neigh, sysp, relay, hosted, n, name, expect_neigh=None,
          expect_sysp=None):
    for a in range(n):
        assert len(set(neigh[a])) == len(neigh[a]), f"{name}: dup neighbour"
        for p in sysp[a]:
            assert p not in neigh[a], f"{name}: {p} both direct and relayed"
        for r in relay[a]:
            assert r in neigh[a], f"{name}: relay {r} not a neighbour of {a}"
        for j, p in enumerate(sysp[a]):
            r = relay[a][j]
            assert p in neigh[r], f"{name}: relay {r} does not wait on {p}"
            assert p in hosted[r], f"{name}: {r} does not re-publish {p}"
        for p in range(n):
            if p == a:
                continue
            assert (p in neigh[a]) or (p in sysp[a]), f"{name}: no route {a}->{p}"
    if expect_neigh is not None:
        for a in range(n):
            assert set(neigh[a]) == set(expect_neigh[a]), \
                f"{name}: rank {a} neighbours {neigh[a]} != {expect_neigh[a]}"
    if expect_sysp is not None:
        for a in range(n):
            assert sysp[a] == expect_sysp[a], \
                f"{name}: rank {a} sysp {sysp[a]} != {expect_sysp[a]}"
    print(f"  {name}: OK  " + " ".join(
        f"r{a}=({len(neigh[a])}n,{len(sysp[a])}s)" for a in range(n)))


def main() -> int:
    # All pairs report peer access on this box (measured), which is exactly the
    # case that must NOT turn SYS pairs into direct links.
    full = [sorted(b for b in range(8) if b != a) for a in range(8)]
    n8, s8, r8, h8 = derive_tables(full, TOP8)
    check(n8, s8, r8, h8, 8, "TP8 (all P2P-capable, SYS pairs)", OLD_N, OLD_S)

    four = [sorted(b for b in range(4) if b != a) for a in range(4)]
    n4, s4, r4, h4 = derive_tables(four, [r[:4] for r in TOP8[:4]])
    check(n4, s4, r4, h4, 4, "TP4 0-3 (all NVLink)", [[1, 2, 3], [0, 2, 3],
                                                      [0, 1, 3], [0, 1, 2]],
          [[], [], [], []])
    n4b, s4b, r4b, h4b = derive_tables(four, [r[4:] for r in TOP8[4:]])
    check(n4b, s4b, r4b, h4b, 4, "TP4 4-7 (all NVLink)", [[1, 2, 3], [0, 2, 3],
                                                          [0, 1, 3], [0, 1, 2]],
          [[], [], [], []])

    # Mixed box: 0-1 NVLink, 0-2 PCIe switch, 0-3 SYS, 2-3 NVLink. The only
    # usable path 0->3 is the PCIe-switch hop 0-2 plus NVLink 2-3.
    mixed = [["X", "NV2", "PXB", "SYS"],
             ["NV2", "X", "PXB", "SYS"],
             ["PXB", "PXB", "X", "NV2"],
             ["SYS", "SYS", "NV2", "X"]]
    usable = [[1, 2], [0, 2], [0, 1, 3], [2]]
    nm, sm, rm, hm = derive_tables(usable, mixed)
    check(nm, sm, rm, hm, 4, "mixed NVLink+PXB+SYS")
    assert rm[0] == [2], f"expected rank 2 as relay for 0->3, got {rm[0]}"

    # One-way peer access (0 cannot reach 1, 1 can reach 0): the pair must be
    # dropped as a direct link on BOTH sides and routed through a rank that
    # reaches both ends.
    one_way = [sorted(b for b in range(4) if b != a) for a in range(4)]
    one_way[0] = [2, 3]
    n1, s1, r1, h1 = derive_tables(one_way, [r[:4] for r in TOP8[:4]])
    check(n1, s1, r1, h1, 4, "one-way pair (routed, not direct)")
    assert 1 not in n1[0] and 0 not in n1[1], "one-way pair used as direct"
    assert 1 in s1[0] and 0 in s1[1], "one-way pair was not routed"

    # A pair that can only be reached over two relays must fail closed.
    two_hop_only = [[1], [0, 2], [1, 3], [2]]
    try:
        derive_tables(two_hop_only, [r[:4] for r in TOP8[:4]])
    except RuntimeError:
        print("  chain topology (needs 2 relays): OK (fails closed)")
    else:
        raise AssertionError("chain topology should have failed closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
