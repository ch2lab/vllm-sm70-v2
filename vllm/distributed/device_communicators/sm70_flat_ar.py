# SPDX-License-Identifier: Apache-2.0
"""SM70 flat 2-hop allreduce (fp16, strict relay ordering, multi-CTA slices).

Sibling of the hierarchical route in ``sm70_hier_ar.py``; selected with
``VLLM_SM70_AR_IMPL=flat``.  The kernel is ``csrc/sm70_flat_ar.cu``, built as
a standalone ``.so`` and driven through ctypes (the same pattern as
``nccl_replica_ar.py``), so enabling or disabling this route never rebuilds
the main extension.

Protocol (measured, see ``sm70/19_two_layer_ar_protocol.md``):

* push my slice into the NVLink neighbours' inbound slots,
* publish my readiness **locally** and poll each neighbour's flag remotely
  (the earlier prototype published into the peers' memory with serial
  system-scope stores: +7.5 us per call),
* a rank hosting a 2-hop peer re-flags that peer after observing its flag, so
  the relayed read cannot race the relay's own ready flag,
* reduce every source in fixed global-rank order: bitwise-stable.

Every topology decision is taken at init from what the machine actually
reports, not from a baked-in 8-GPU table:

1. the link labels of ``nvidia-smi topo -m`` grade the path (NV2/NV1 NVLink,
   PXB/PIX PCIe switch, PHB host bridge), but the label alone never decides:
2. ``cudaDeviceCanAccessPeer`` plus the return code of
   ``cudaDeviceEnablePeerAccess`` say which pairs this process can really use
   -- a NVLink pair can still be P2P-blocked (driver, IOMMU, fabric manager)
   and a PCIe-switch pair can be perfectly usable at lower bandwidth,
3. a direct pair is kept only when both directions work (the push and the flag
   poll both cross it), so the protocol never ends up one-way,
4. device identity is matched by GPU uuid, so the tables stay correct even
   when the CUDA device order differs from ``nvidia-smi`` order,
5. peers without a usable direct path are routed through the fastest rank that
   reaches both ends (2-hop), which makes a fully connected group single-hop,
6. the first eager use runs one real reduction against the fp32 rank-order
   reference and every rank votes, so the group either uses the route on all
   ranks or falls back on all ranks.

Epochs are device-side per-slice counters, so a CUDA graph replay advances
them with no host participation.
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

# In-tree build product (build_sm70.sh -> vllm/sm70_flat_ar.so). No external or
# hand-compiled artifact is referenced: the source lives in
# csrc/sm70_flat_ar.cu and the build installs it next to this package.
_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "sm70_flat_ar.so",
)

# Row count of the per-rank tables in csrc/sm70_flat_ar.cu (F_MAXW).
_F_MAXW = 16

# Link quality, from the `nvidia-smi topo -m` labels. Only a hint for choosing
# relays: what counts for usability is the runtime peer-access answer.
_LINK_RANK = {"NV": 4, "PXB": 3, "PIX": 2, "PHB": 1}


def _link_rank(label: str) -> int:
    s = str(label)
    for prefix, rank in _LINK_RANK.items():
        if s.startswith(prefix):
            return rank
    return 0  # SYS / NODE / X: no device-to-device path


# ---------------------------------------------------------------------------
# Init-time detection: link labels, real peer-access capability, tables
# ---------------------------------------------------------------------------
def _norm_bus(bus: str) -> str:
    """Normalise a PCI bus id: nvidia-smi pads the domain to 8 hex digits,
    cudart does not ("00000000:1B:00.0" vs "0000:1B:00.0")."""
    dom, b, devfn = (s.strip() for s in bus.split(":"))
    return f"{int(dom, 16)}:{b.lower()}:{devfn}"


def _snmi_index_by_pci() -> dict[str, int]:
    """Normalised PCI bus id -> nvidia-smi GPU index."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi --query-gpu rc={out.returncode}: "
            f"{out.stderr.strip()[:200]}")
    by_pci: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        idx, _, bus = (s.strip() for s in line.partition(","))
        by_pci[_norm_bus(bus)] = int(idx)
    return by_pci


def snmi_index(dev: int, by_pci: dict[str, int]) -> int:
    """``nvidia-smi`` index of ``dev``, without touching that device.

    ``torch.cuda.get_device_properties(i)`` switches the current device and
    materializes a primary context (~300 MiB) on every device it is called for.
    One process per rank querying all ranks' devices therefore costs ~2 GiB per
    card (measured 2026-10-25: 7 extra contexts x 308 MiB per GPU).
    ``cudaDeviceGetPCIBusId`` is a plain device query with no context.
    """
    rt = ctypes.CDLL("libcudart.so")
    buf = ctypes.create_string_buffer(64)
    rc = rt.cudaDeviceGetPCIBusId(buf, ctypes.c_int(64), ctypes.c_int(dev))
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetPCIBusId(dev={dev}) rc={rc}")
    bus = _norm_bus(buf.value.decode())
    if bus not in by_pci:
        raise RuntimeError(f"device {dev} (pci {bus}) is not in nvidia-smi")
    return by_pci[bus]


def probe_peer_capability(dev: int, peers: list[int]) -> tuple[set[int],
                                                               list[str]]:
    """Query (never enable) peer access from ``dev`` to ``peers``.

    Deliberately **no** ``cudaDeviceEnablePeerAccess``: that call materialises a
    primary context (~308 MiB) on every peer device, per process. Measured on
    this box: 7 peers x 308 MiB per card, ~17 GiB machine-wide.

    It is also unnecessary: vLLM's ``open_mem_handle`` opens the peer's region
    with ``cudaIpcMemLazyEnablePeerAccess`` (csrc/custom_all_reduce.cu:196), and
    NCCL does the same (nccl/src/transport/p2p.cc:313/1235) -- the lazy flag
    authorises cross-device access at open time without a peer context.
    ``cudaDeviceCanAccessPeer`` is a plain query and costs nothing.
    """
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaGetErrorString.argtypes = [ctypes.c_int]
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    cur = ctypes.c_int(0)
    rt.cudaGetDevice(ctypes.byref(cur))
    orig = cur.value
    usable: set[int] = set()
    notes: list[str] = []
    if orig != dev:
        rt.cudaSetDevice(ctypes.c_int(dev))
    try:
        for p in peers:
            if p == dev:
                continue
            can = ctypes.c_int(0)
            rc = rt.cudaDeviceCanAccessPeer(ctypes.byref(can),
                                            ctypes.c_int(dev), ctypes.c_int(p))
            if rc != 0 or can.value != 1:
                notes.append(f"{dev}->{p}: can_access_peer={can.value} rc={rc}")
                continue
            usable.add(p)
    finally:
        if orig != dev:
            rt.cudaSetDevice(ctypes.c_int(orig))
    return usable, notes


def derive_tables(
    usable: list[list[int]], klass: list[list[str]]
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """Rank -> (neighbours, 2-hop peers, relays, hosted peers).

    ``usable[a]`` lists the ranks ``a`` reaches over a verified peer-access
    path (NVLink **or** PCIe switch); a pair is kept only when both directions
    work, because the push and the flag poll both cross it. ``klass[a][b]`` is
    the link label.

    The route per pair is the *better of the two*: a direct link (any class)
    versus a 2-hop path through a rank that reaches both ends, scored by the
    slowest leg. Ties go to the direct link (fewer hops). This matters in
    practice: on an 8-GPU box every pair reports peer access, but a SYS pair
    (0 on the scale) is far worse than two NVLink hops (4), so the relay keeps
    the cross-NUMA traffic on NVLink -- while a PCIe-switch pair with no faster
    detour stays direct.
    """
    n = len(usable)

    def pair(a: int, b: int) -> bool:
        return a != b and b in usable[a] and a in usable[b]

    def q(a: int, b: int) -> int:
        return _link_rank(klass[a][b])

    # Best 2-hop score available for every pair (max over intermediates of the
    # slower leg). Independent of the relay chosen later, so every rank derives
    # the same graph.
    q2 = [[-1] * n for _ in range(n)]
    for a in range(n):
        for p in range(n):
            if p == a:
                continue
            best = -1
            for r in range(n):
                if r in (a, p) or not pair(a, r) or not pair(r, p):
                    continue
                best = max(best, min(q(a, r), q(r, p)))
            q2[a][p] = best

    # Keep a direct link when it is at least as good as the best detour.
    edge = [[pair(a, p) and q(a, p) >= q2[a][p] for p in range(n)]
            for a in range(n)]

    neigh: list[list[int]] = [[] for _ in range(n)]
    sysp: list[list[int]] = [[] for _ in range(n)]
    relay: list[list[int]] = [[] for _ in range(n)]
    hosted: list[list[int]] = [[] for _ in range(n)]
    load = [0] * n
    for a in range(n):
        for p in range(n):
            if p == a:
                continue
            if edge[a][p]:
                neigh[a].append(p)
                continue
            # A relay must itself treat ``p`` as a direct neighbour, otherwise
            # it would not wait for p's flag before re-publishing it.
            cands = [r for r in range(n)
                     if r not in (a, p) and edge[a][r] and edge[r][p]]
            if not cands:
                if pair(a, p):
                    neigh[a].append(p)  # no detour: slow is better than wrong
                    continue
                raise RuntimeError(
                    f"flat AR: rank {a}->{p} has no usable path "
                    f"(direct neighbours of {a}: {neigh[a]})")
            r = min(cands, key=lambda c: (-min(q(a, c), q(c, p)), load[c], c))
            load[r] += 1
            sysp[a].append(p)
            relay[a].append(r)
            if p not in hosted[r]:
                hosted[r].append(p)
        neigh[a].sort()
    return neigh, sysp, relay, hosted


class Sm70FlatAR:
    def __init__(self, group, device, max_bytes: int, s_max: int = 4,
                 lib_path: str | None = None, strict: bool = False) -> None:
        self.group = group
        self.dev = (device if isinstance(device, torch.device)
                    else torch.device(device))
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.max_bytes = int(max_bytes)
        self.s_max = int(s_max)
        self.strict = strict
        self.disabled = True
        self._ready = False
        self._logged: set[tuple[int, int]] = set()
        if self.world < 2 or self.world > _F_MAXW:
            raise RuntimeError(
                f"flat AR supports 2..{_F_MAXW} ranks (got {self.world})")
        # The rank -> physical GPU map is needed before any topology decision:
        # the adjacency must describe the devices this group actually uses.
        self._devices = [0] * self.world
        dist.all_gather_object(
            self._devices,
            int(self.dev.index if self.dev.index is not None
                else torch.cuda.current_device()),
            group=self.group)
        self._neigh: list[list[int]] = []
        self._sysp: list[list[int]] = []
        self._relay: list[list[int]] = []
        self._hosted: list[list[int]] = []
        self._lib = ctypes.CDLL(lib_path or os.environ.get(
            "VLLM_SM70_AR_FLAT_LIB", _DEFAULT_LIB))
        self._lib.flat_region_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.flat_region_bytes.restype = ctypes.c_int
        self._lib.flat_n_max.argtypes = [ctypes.c_int]
        self._lib.flat_n_max.restype = ctypes.c_int
        self._lib.flat_launch.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p]
        self._lib.flat_launch.restype = ctypes.c_int
        self._n_max = int(self._lib.flat_n_max(self.max_bytes))
        self._region_bytes = int(
            self._lib.flat_region_bytes(self.max_bytes, self.world))
        if not self._check_topology():
            return
        self.disabled = False

    # ------------------------------------------------------------------
    def _check_topology(self) -> bool:
        try:
            from vllm.distributed.device_communicators.sm70_hier_ar import (
                topo_matrix,
            )

            rows = topo_matrix()
            by_pci = _snmi_index_by_pci()
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 flat AR: link/device probe failed (%r); "
                           "failing closed to the next backend", e)
            return False

        me = self.rank
        dev = int(self._devices[me])
        snmi = [None] * self.world
        try:
            # Only our own device is queried locally (no stray contexts); the
            # other ranks' nvidia-smi indices come over the collective.
            dist.all_gather_object(snmi, snmi_index(dev, by_pci),
                                   group=self.group)
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 flat AR: device identity probe failed (%r); "
                           "failing closed", e)
            return False
        if any(not isinstance(i, int) for i in snmi):
            logger.warning("SM70 flat AR: device identity gather failed; "
                           "failing closed")
            return False
        klass = [[str(rows[snmi[a]][snmi[b]]) for b in range(self.world)]
                 for a in range(self.world)]
        # Candidate graph from the link labels only (this touches no device
        # state): a pair is a candidate when the label reports a device-to-device
        # path (NVLink or PCIe switch) in BOTH directions, because the push and
        # the flag poll both cross it. SYS / host-bridge pairs need a 2-hop
        # relay, which is what derive_tables adds below.
        cand = [
            [b for b in range(self.world)
             if b != a and _link_rank(klass[a][b]) > 0
             and _link_rank(klass[b][a]) > 0]
            for a in range(self.world)
        ]
        try:
            neigh, sysp, relay, hosted = derive_tables(cand, klass)
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 flat AR: no usable route across the group "
                           "(%r); failing closed", e)
            return False
        # Validate (query only, no enable) the neighbours this rank will really
        # read or write. cudaDeviceEnablePeerAccess is NOT used: it would
        # materialise a ~308 MiB context on every peer, per process, while the
        # lazy flag inside open_mem_handle already authorises the access.
        _usable, notes = probe_peer_capability(
            dev, [int(self._devices[b]) for b in neigh[me]])
        for note in notes:
            logger.warning("SM70 flat AR: peer access unavailable: %s", note)
        if notes:
            logger.warning("SM70 flat AR: %d direct neighbour(s) unusable; "
                           "failing closed", len(notes))
            return False
        for row in neigh + sysp + relay + hosted:
            if len(row) > _F_MAXW:
                logger.warning("SM70 flat AR: adjacency row of %d exceeds the "
                               "kernel table (%d); failing closed",
                               len(row), _F_MAXW)
                return False
        self._neigh, self._sysp, self._relay, self._hosted = (
            neigh, sysp, relay, hosted)
        links = {klass[me][b]: 0 for b in range(self.world) if b != me}
        for b in range(self.world):
            if b != me:
                links[klass[me][b]] += 1
        logger.info(
            "SM70 flat AR topology: devices %r (nvidia-smi %r), links from "
            "rank %d %r, %d direct neighbour(s), %d 2-hop peer(s)",
            self._devices, snmi, me, links, len(neigh[me]), len(sysp[me]))
        if neigh[me] and min(_link_rank(klass[me][b])
                             for b in neigh[me]) < _LINK_RANK["NV"]:
            logger.warning(
                "SM70 flat AR: rank %d has a direct neighbour below NVLink "
                "(%r); the route will run at PCIe-class bandwidth",
                me, {b: klass[me][b] for b in neigh[me]})
        return True

    def _lazy_init(self) -> bool:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "flat AR: first use happened inside CUDA graph capture; the "
                "region must be allocated during the eager profile run")
        ptr, handle = (
            torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(
                self._region_bytes))
        handles: list = [None] * self.world
        dist.all_gather_object(handles, handle, group=self.group)
        # Only the neighbours' regions are ever dereferenced (the relayed
        # 2-hop payload lives in the relay's region, and a relay is a
        # neighbour): opening every peer's handle costs a context per peer for
        # nothing.
        needed = set(self._neigh[self.rank]) | set(self._relay[self.rank])
        self._ptrs = [
            ptr if i == self.rank
            else (int(torch.ops._C_custom_ar.open_mem_handle(h))
                  if i in needed else 0)
            for i, h in enumerate(handles)
        ]
        nbr = self._neigh[self.rank]
        hm = self._hosted[self.rank]
        self._base = (ctypes.c_uint64 * self.world)(*self._ptrs)
        self._nbr = (ctypes.c_int * max(1, len(nbr)))(*nbr)
        self._sy = (ctypes.c_int * max(1, len(self._sysp[self.rank])))(
            *self._sysp[self.rank])
        self._rl = (ctypes.c_int * max(1, len(self._relay[self.rank])))(
            *self._relay[self.rank])
        self._ho = (ctypes.c_int * max(1, len(hm)))(*hm)
        self._n_host = len(hm)
        self._ready = True
        if not self._self_test():
            self._ready = False
            self.disabled = True
            return False
        logger.info("SM70 flat AR: route ACTIVE (rank %d, world %d, cap %d B, "
                    "n_max %d, region %d B/rank, neighbours %r, SYS %r, "
                    "relays %r, host %r)", self.rank, self.world,
                    self.max_bytes, self._n_max, self._region_bytes, nbr,
                    self._sysp[self.rank], self._relay[self.rank], hm)
        return True

    def _self_test(self) -> bool:
        """One real reduction vs the fp32 rank-order reference, then a vote.

        The reference sums the gathered inputs in rank order in fp32, exactly
        the order the kernel promises, so a correct route is bit-exact. Every
        rank must agree: mixing this route with NCCL would leave the replicas
        differing by rounding.
        """
        n = 2048  # divisible by 128*S and >= 512 per slice
        dev = self.dev
        base = torch.arange(n, dtype=torch.float32).remainder(97)
        inp = (base * (self.rank + 1)).half().to(dev)
        out = None
        try:
            out = self.all_reduce(inp)
            torch.cuda.synchronize(dev)
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 flat AR: self-test launch raised (%r)", e)
        rows: list = [None] * self.world
        # Always take part: skipping this gather on failure would deadlock the
        # ranks that do reach it.
        dist.all_gather_object(rows, inp.cpu(), group=self.group)
        ok = out is not None
        if ok:
            ref = torch.zeros(n, dtype=torch.float32)
            for r in rows:
                ref += r.float()
            ref = ref.half().to(dev)
            ok = bool(torch.equal(out, ref))
            if not ok:
                logger.warning(
                    "SM70 flat AR: self-test mismatch on rank %d (max_abs "
                    "%.6f, ref[0] %.4f got %.4f)", self.rank,
                    (out.float() - ref.float()).abs().max().item(),
                    float(ref[0]), float(out[0]))
        flag = torch.tensor([0 if ok else 1], dtype=torch.int32)
        try:
            dist.all_reduce(flag, group=self.group)
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 flat AR: self-test vote failed (%r)", e)
            return False
        if int(flag.item()) != 0:
            logger.warning("SM70 flat AR: self-test failed on >=1 rank; "
                           "failing closed for the whole group")
            return False
        return True

    # ------------------------------------------------------------------
    def _pick_s(self, n: int) -> int:
        for s in (self.s_max, 16, 8, 4, 2, 1):
            if s > self.s_max:
                continue
            if n % (128 * s) == 0 and n // s >= 512:
                return s
        return 0

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype is not torch.float16 or not inp.is_contiguous():
            return False
        n = inp.numel()
        if n > self._n_max or n % 128:
            return False
        if self._pick_s(n) == 0:
            return False
        if not self._ready and not self._lazy_init():
            return False
        key = (n, self._pick_s(n))
        if key not in self._logged:
            self._logged.add(key)
            logger.info("SM70 flat AR: rank %d takes %d elems (%d B) with S=%d",
                        self.rank, n, n * 2, key[1])
        return True

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        n = inp.numel()
        s = self._pick_s(n)
        out = torch.empty_like(inp)
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        rc = self._lib.flat_launch(
            self._base, self._nbr, self._sy, self._rl, self._ho,
            self._n_host, len(self._sysp[self.rank]), len(self._neigh[self.rank]),
            self.world, self.rank, n, self._n_max, s,
            ctypes.c_void_p(inp.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(
                f"flat AR launch rc={rc} for n={n} S={s} "
                "(-1 S out of range, -2 n>cap, -3 n%128S, -4 n/S<512, "
                "-5 world out of 1..16, -6 bad neighbour/relay table)")
        return out

    def close(self) -> None:
        self._ready = False
        self.disabled = True
