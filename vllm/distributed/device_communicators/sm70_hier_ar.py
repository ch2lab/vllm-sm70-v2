# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 topology-aware hierarchical all-reduce — round-3 AR shadow route.

Decode/verify-size AR ONLY (the prefill flat-AR leg is NOT hijacked; spec §3
eligibility). Default-off (rule 11670): with ``VLLM_SM70_AR_HIER=0`` and no
control file, the dispatch hook in ``CudaCommunicator.all_reduce`` is a single
None-check — no allocation, no probe, no file I/O, no kernel load.

Idea-level sources (rule 11671 — nothing copied from ``/data/src/1cat-vllm``;
the protocol below is a new-tree-native re-implementation of that idea class):

- 1Cat ``sm70_deepseek_v4_tp8_graph_allreduce.md``:
  * hierarchical 2-clique protocol (4-way intra-clique reduce -> FP32 clique
    partial in fixed global-rank order; mirror-pair cross-clique exchange
    over the single cross-quad NVLink leg; one final FP32 add + single FP16
    downcast -> bitwise-equal-across-ranks);
  * double-buffered graph-safe handshakes (clique-ready, pair-ready AND data
    slots double-buffered; local-input consumption acknowledgement published
    concurrently with the pair exchange — the 1Cat accepted protocol, none
    of its rejected variants);
  * device-side waits replacing the collective graph-launch coordination
    (the graph-scheduling win, spec §2.d);
  * the sidecar-library override env (1Cat ``VLLM_SM70_CUSTOM_AR_LIBRARY``)
    -> ``VLLM_SM70_AR_LIB`` (bench-only).
- House patterns (this tree): ``custom_all_reduce.py``
  ``create_shared_buffer`` (IPC handle allgather + open),
  ``csrc/custom_collective_common.cuh`` release/acquire flag idioms, and the
  R2-T1b control-file read pattern (``_read_qsa_warps_file`` — read per
  launch when set, missing/garbage falls back to the static env value,
  warn at most once; that seam was promoted in-place by
  commit 8bee367a93 and this module re-establishes the pattern for the AR
  seam per spec §3).

Topology (boot-verified; fail-closed on any mismatch — spec §6 risk R1):
  quads {0,1,2,3} / {4,5,6,7} = full 4-GPU NVLink cliques
  mirror legs (the only cross-quad NVLinks): 0<->7, 1<->6, 2<->5, 3<->4
  Source of truth: the LINK-LEVEL matrix (nvidia-smi topo -m, NV# vs
  SYS) — the driver's P2P capability query alone cannot classify the
  16 NVLink / 12 SYS pairs (it reports PCIe-P2P True on some SYS pairs;
  measured). See topo_matrix() + _check_topology.

Contract: ``.superpowers/sdd/2026-10-13-sm70-round3-ar/ar-direction-spec.md``.
"""

from __future__ import annotations

import os
import re
import subprocess

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_AR_TRACE = os.environ.get("VLLM_SM70_AR_TRACE", "0") == "1"
_AR_REJECT_LOG = os.environ.get("VLLM_SM70_AR_REJECT_LOG", "0") == "1"
_AR_CALLERS: dict[str, int] = {}
_AR_CALLS = 0

# Per-call adaptive route/wire policy (2026-10-22). Entries are measured, not
# guessed:
#   decode / small (<= max_bytes): latency-bound, 5-15 KB payloads whose copy is
#     <1 us of a ~26 us call -> fp16 (wire=3, near-lossless) is optimal; e4m3 and
#     nvfp4 buy no time here and were rejected on precision (2026-10-18).
#   prefill / large (> max_bytes): bandwidth-bound MB messages. NCCL's ring is the
#     throughput route (98-113 GB/s); forcing them onto the clique kernel with a
#     compressed wire measured -48% prefill throughput (2026-10-22). The ring
#     route (AR_RING_PREFILL) stays behind its own valve and is not licensed yet.
# The table exists so the per-call decision is explicit and auditable; with the
# currently licensed routes it resolves to today's production behaviour.
_AR_ADAPTIVE = os.environ.get("VLLM_SM70_AR_ADAPTIVE", "0") == "1"
# Diagnostic bypass (2026-10-22): return the local partial (numerically wrong,
# structurally identical) to measure AR's *marginal* cost by A/B. Default off.
_AR_BYPASS = os.environ.get("VLLM_SM70_AR_BYPASS", "0") == "1"


# --- AR telemetry -----------------------------------------------------------
# Counters live in HOST-MAPPED pinned memory (sm70_hier_ar.cu): reading them is
# a plain ctypes load with no CUDA call, no stream and no synchronisation, which
# is what makes polling legal while the engine captures CUDA graphs.
#   per-wire [0..16)  {calls, total_ns, total_bytes, max_ns} x wire mode
#   per-hop  [16..24) {calls, total_ns, max_ns} x {quad, mirror}
_AR_STATS_LOG = os.environ.get("VLLM_SM70_AR_STATS_LOG")
_AR_STATS_INTERVAL = float(os.environ.get("VLLM_SM70_AR_STATS_INTERVAL", "5"))
_AR_STATS_THREAD: "threading.Thread | None" = None
_AR_STATS_PTR = 0
_WIRE_NAMES = ("fp32", "e4m3", "nvfp4", "fp16")
# The prefill-sized collectives run on the NCCL ring, which has no in-kernel
# counters, so their load is accumulated here (plain Python ints; the sampler
# thread reads them, no CUDA involvement at all).
_RING_STATS = {"calls": 0, "bytes": 0}
_HOP_NAMES = ("quad", "mirror")


def set_ar_stats_ptr(ptr: int) -> None:
    """Publish the host-mapped counter address (called at FA init)."""
    global _AR_STATS_PTR
    _AR_STATS_PTR = int(ptr or 0)


def read_host_stats(ptr: int | None = None) -> dict:
    """Read the 24 host-mapped counters through plain memory."""
    import ctypes

    addr = _AR_STATS_PTR if ptr is None else int(ptr)
    if not addr:
        return {"wire": [], "hop": []}
    raw = list((ctypes.c_uint64 * 24).from_address(addr))
    wire = []
    for w, name in enumerate(_WIRE_NAMES):
        calls, ns, nbytes, mx = raw[w * 4 : w * 4 + 4]
        wire.append(
            {
                "wire": name,
                "calls": calls,
                "mean_us": (ns / calls / 1000.0) if calls else 0.0,
                "max_us": mx / 1000.0,
                "mb": nbytes / 1e6,
                "gbps": (nbytes / ns) if ns else 0.0,
            }
        )
    hop = []
    for h, name in enumerate(_HOP_NAMES):
        calls, ns, mx = raw[16 + h * 4 : 16 + h * 4 + 3]
        hop.append(
            {
                "hop": name,
                "calls": calls,
                "mean_us": (ns / calls / 1000.0) if calls else 0.0,
                "max_us": mx / 1000.0,
            }
        )
    return {"wire": wire, "hop": hop}


def ar_stats() -> dict:
    """Per-wire/per-hop AR counters (host-mapped; capture-safe)."""
    return read_host_stats()


def _stats_logger_loop(path: str, interval: float) -> None:
    import json
    import time

    prev = None
    prev_ring = (0, 0)
    while True:
        time.sleep(interval)
        try:
            cur = read_host_stats()["wire"]
            if prev is not None:
                delta = []
                for c, p in zip(cur, prev):
                    calls = c["calls"] - p["calls"]
                    ns = (
                        c["mean_us"] * c["calls"] - p["mean_us"] * p["calls"]
                    ) * 1000.0
                    mb = c["mb"] - p["mb"]
                    delta.append(
                        {
                            "wire": c["wire"],
                            "calls": calls,
                            "mean_us": (ns / calls / 1000.0) if calls else 0.0,
                            "mb": mb,
                            "gbps": (mb * 1e6 / ns) if ns > 0 else 0.0,
                        }
                    )
                rcalls = _RING_STATS["calls"] - prev_ring[0]
                rbytes = _RING_STATS["bytes"] - prev_ring[1]
                rec = {
                    "ts": time.time(),
                    "interval_s": interval,
                    "ar": delta,
                    "ring": {"calls": rcalls, "bytes": rbytes},
                }
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            prev = cur
            prev_ring = (_RING_STATS["calls"], _RING_STATS["bytes"])
        except Exception:  # noqa: BLE001 - telemetry must never break serving
            prev = None


def start_ar_stats_logger() -> None:
    """Start the JSONL sampler once, if VLLM_SM70_AR_STATS_LOG is set."""
    global _AR_STATS_THREAD
    if not _AR_STATS_LOG or _AR_STATS_THREAD is not None or not _AR_STATS_PTR:
        return
    import threading

    _AR_STATS_THREAD = threading.Thread(
        target=_stats_logger_loop,
        args=(_AR_STATS_LOG, _AR_STATS_INTERVAL),
        name="sm70-ar-stats",
        daemon=True,
    )
    _AR_STATS_THREAD.start()


def _choose_route_wire(phase: str, nbytes: int, max_bytes: int) -> tuple[str, int]:
    """(route, wire) for one collective; wire in {0 fp32, 1 e4m3, 2 nvfp4, 3 fp16}."""
    if phase == "decode" or nbytes <= max_bytes:
        return "hier", 3
    return "nccl", 3



def _ar_trace_note() -> None:
    """Diagnostic (default off): who issues each all-reduce, aggregated."""
    global _AR_CALLS
    import json
    import traceback

    frames = traceback.extract_stack(limit=16)[:-1]
    key = " <- ".join(
        f"{f.filename.rsplit('/', 1)[-1]}:{f.name}" for f in frames[-12:]
    )
    _AR_CALLERS[key] = _AR_CALLERS.get(key, 0) + 1
    _AR_CALLS += 1
    if _AR_CALLS % 200 == 0:
        try:
            with open(f"/tmp/sm70_ar_callers_{os.getpid()}.json", "w") as fh:
                json.dump(_AR_CALLERS, fh, indent=1)
        except OSError:
            pass

# The 16 NVLink pairs the route needs (spec §1.1: 8 NV2 + 8 NV1; the other
# 12 of 28 pairs are SYS = cross-NUMA PCIe, HW_GUIDE §3.1 measured).
# NOTE: the driver's P2P capability query reports True on SOME SYS pairs
# (measured 2026-10-13, ar_smoke4: (0,4)/(0,5)/(0,6)) — it cannot classify
# the matrix. The LINK-LEVEL matrix (topo_matrix) is the classification
# source; the capability query is used only as the 16-leg health pre-
# filter in _check_topology (SYS pairs are unconstrained).
#   intra-quad: quad A {0,1,2,3} + quad B {4,5,6,7}, all 6 pairs each
#   mirror legs (the only cross-quad NVLinks): 0<->7, 1<->6, 2<->5, 3<->4
_EXPECTED_P2P_PAIRS: frozenset[frozenset[int]] = frozenset(
    frozenset(p)
    for p in (
        # intra-quad (12): NV2 (0,1)(0,2)(1,3)(4,6)(5,7)(6,7)
        # + NV1 (0,3)(1,2)(2,3)(4,5)(4,7)(5,6)
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
        # mirror legs (4): NV1 (0,7)(1,6) + NV2 (2,5)(3,4)
        (0, 7), (1, 6), (2, 5), (3, 4),
    )
)
# The production prefill point (spec §1.2: M=8192, hidden 2560, fp16) — the
# mode-2 ceiling.
_PREFILL_MAX_BYTES = 8192 * 2560 * 2  # 41,943,040 B

# ------------------------------------------------------------------
# Link-level topology (the design's pair-classification input)
# ------------------------------------------------------------------
# The driver's P2P capability query (torch.cuda.can_device_access_peer)
# cannot discriminate the 16 NVLink pairs from the 12 SYS pairs on this
# box: it reports True for PCIe-P2P-capable SYS pairs too (measured
# 2026-10-13, ar_smoke4: (0,4)/(0,5)/(0,6) P2P=True, nvidia-smi topo -m
# says SYS). Only LINK-LEVEL data discriminates them: the nvidia-smi
# topo -m matrix (NV# = a bonded set of # NVLinks, SYS = cross-NUMA
# PCIe). Hence the boot topology check is:
#   (a) the matrix's NV-labeled pair set == _EXPECTED_P2P_PAIRS exactly
#       (a rewire shows up here as a missing/extra NV pair),
#   (b) every NV pair is P2P-accessible (the capability query, used only
#       as the 16-leg health pre-filter),
#   (c) SYS pairs carry NO constraint — the route never uses them.
# Transfer-level verification is the two-phase IPC readback below.
# Both helpers are module-level (sm70_tests/selfcheck_ar_hier.py mocks
# them; the selfcheck never runs the subprocesses).

_TOPO_MATRIX_CACHE: list[list[str]] | None = None
_PHYS_IDX_CACHE: list[int] | None = None
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def topo_matrix() -> list[list[str]]:
    """LINK-LEVEL pair matrix from `nvidia-smi topo -m`, indexed by
    PHYSICAL gpu index (the `nvidia-smi -L` enumeration): cell = link
    label ("X", "NV1", "NV2", "SYS", "PXB", ...). Cached per process
    (the boot probe runs once per rank). Raises on ANY parse failure —
    the caller (``_check_topology``) fails the rank closed."""
    global _TOPO_MATRIX_CACHE
    if _TOPO_MATRIX_CACHE is not None:
        return _TOPO_MATRIX_CACHE
    out = subprocess.run(["nvidia-smi", "topo", "-m"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi topo -m rc={out.returncode}: "
            f"{out.stderr.strip()[:200]}")
    lines = _ANSI_RE.sub("", out.stdout).splitlines()
    header = next((l for l in lines if l[:1].isspace()
                   and re.search(r"GPU\d+", l)), None)
    if header is None:
        raise RuntimeError("topo -m: header row not found")
    cells = header.split()
    n = 0
    while n < len(cells):
        m = re.fullmatch(r"GPU(\d+)", cells[n])
        if not m or int(m.group(1)) != n:
            break
        n += 1
    if n == 0:
        raise RuntimeError("topo -m: no GPU columns in the header")
    parsed: dict[int, list[str]] = {}
    for l in lines:
        if not l.startswith("GPU"):
            continue
        cells = l.split()
        m = re.fullmatch(r"GPU(\d+)", cells[0])
        if not m:
            raise RuntimeError(
                f"topo -m: unparseable row label {cells[0]!r}")
        idx = int(m.group(1))
        if idx in parsed:
            raise RuntimeError(f"topo -m: duplicate row GPU{idx}")
        parsed[idx] = cells[1:1 + n]
    if sorted(parsed) != list(range(n)):
        raise RuntimeError(
            f"topo -m: rows {sorted(parsed)} != GPU0..GPU{n - 1}")
    known = {"X", "SYS", "NODE", "PHB", "PXB", "PIX"}
    matrix: list[list[str]] = [[""] * n for _ in range(n)]
    for i in range(n):
        row = parsed[i]
        if len(row) < n:
            raise RuntimeError(
                f"topo -m: row GPU{i} has {len(row)} link cells, "
                f"expected {n}")
        for j in range(n):
            lab = row[j]
            if not (lab.startswith("NV") or lab in known):
                raise RuntimeError(
                    f"topo -m: unknown link label {lab!r} at ({i},{j})")
            matrix[i][j] = lab
    for i in range(n):
        if matrix[i][i] != "X":
            raise RuntimeError(
                f"topo -m: diagonal ({i},{i}) = {matrix[i][i]!r}, "
                "expected X")
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                raise RuntimeError(
                    f"topo -m: asymmetric at ({i},{j}): "
                    f"{matrix[i][j]!r} vs {matrix[j][i]!r}")
    _TOPO_MATRIX_CACHE = matrix
    return matrix


def visible_ordinal_to_physical(n_visible: int) -> list[int]:
    """Physical index (the `nvidia-smi -L` enumeration) of each torch-
    visible ordinal 0..n_visible-1, so the physical-indexed link matrix
    lines up with the ordinal-indexed P2P capability queries. Identity
    when all gpus are visible (gate-1 bench, the TP8 service); UUID-
    joined when a visible subset is set. Raises on any mismatch — the
    caller fails the rank closed."""
    global _PHYS_IDX_CACHE
    if _PHYS_IDX_CACHE is not None and len(_PHYS_IDX_CACHE) == n_visible:
        return _PHYS_IDX_CACHE
    out = subprocess.run(["nvidia-smi", "-L"],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi -L rc={out.returncode}")
    phys: dict[str, int] = {}
    for line in out.stdout.splitlines():
        m = re.match(r"GPU (\d+):.*\(UUID: (.*?)\)", line.strip())
        if m:
            phys[m.group(2).removeprefix("GPU-").lower()] = int(m.group(1))
    idx: list[int] = []
    for i in range(n_visible):
        uuid = str(torch.cuda.get_device_properties(i).uuid).lower()
        if uuid not in phys:
            raise RuntimeError(
                f"visible ordinal {i} (UUID {uuid}) not found in "
                "nvidia-smi -L")
        idx.append(phys[uuid])
    _PHYS_IDX_CACHE = idx
    return idx


def _read_mode_file(path: str, default: int) -> int:
    """Runtime-flip control-file read (R2-T1b pattern, house: the removed
    ``_read_qsa_warps_file`` seam): the file's content (a single int) sets
    the route mode for THIS launch only; the controller flips it by an
    atomic ``os.replace`` write. Any error (missing, empty, non-int) falls
    back to ``default``, warning at most once so a persistently missing
    file does not spam the log."""
    try:
        with open(path, encoding="ascii") as f:
            return int(f.read().strip())
    except Exception:  # noqa: BLE001 - any read/parse error -> the default
        global _AR_HIER_FILE_WARNED
        if not _AR_HIER_FILE_WARNED:
            _AR_HIER_FILE_WARNED = True
            logger.warning(
                "VLLM_SM70_AR_HIER_FILE=%s unreadable or not an int — "
                "using mode=%d", path, default)
        return default


_AR_HIER_FILE_WARNED = False

# Interleaved A/B control (2026-10-22): read at most once a second so the arm can
# be flipped inside ONE boot -- cross-boot comparisons carry a 17% systematic
# drift (measured), so an A/B must never straddle a reboot.
_ROUTE_FILE = os.environ.get("VLLM_SM70_AR_ROUTE_FILE",
                             "/root/run/sm70/ar_route_mode")
_ROUTE_CACHE = {"t": 0.0, "v": 1}
_ROUTE_LOGGED: set[tuple[str, str]] = set()


def _route_enabled() -> bool:
    import time as _time

    now = _time.monotonic()
    if now - _ROUTE_CACHE["t"] > 1.0:
        _ROUTE_CACHE["t"] = now
        try:
            with open(_ROUTE_FILE, encoding="ascii") as fh:
                _ROUTE_CACHE["v"] = int(fh.read().strip())
        except Exception:  # noqa: BLE001 - missing/garbage -> arm on (production)
            _ROUTE_CACHE["v"] = 1
    return _ROUTE_CACHE["v"] == 1


def _route_note(phase: str, nbytes: int, route: str, wire: int) -> None:
    key = (phase, route)
    if key in _ROUTE_LOGGED:
        return
    _ROUTE_LOGGED.add(key)
    logger.info("SM70AR route: phase=%s bytes=%d route=%s wire=%d",
                phase, nbytes, route, wire)


# TEMP-AB (2026-10-21, VLLM_SM70_AR_HIER_MAX_BYTES A/B window): hijack-size
# census, log-only, behind VLLM_SM70_AR_HIER_TRACE=1. REMOVE after the window.
_AR_TRACE_HITS: dict[str, int] = {}


def _trace_take(tag: str, nbytes: int, cap: int) -> None:
    import os

    if os.environ.get("VLLM_SM70_AR_HIER_TRACE") != "1":
        return
    key = f"{tag}:{nbytes}"
    c = _AR_TRACE_HITS.get(key, 0) + 1
    _AR_TRACE_HITS[key] = c
    if c in (1, 16, 256, 4096):
        logger.info(
            "SM70 hier AR TRACE: %s n=%d B cap=%d B (hits=%d)",
            tag, nbytes, cap, c)


class Sm70HierAR:
    """The shadow route object: boot-time topology/IPC probe (fail-closed),
    IPC buffers, mode resolution, and the per-call dispatch decision."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        mode_env: int,
        file_path: str,
        max_bytes: int,
        wire: int,
        lib_path: str,
    ) -> None:
        self.group = group
        self.device = device
        self.disabled = True  # fail-closed until the probe passes
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_bytes = max_bytes
        self.mode_env = mode_env
        self.file = file_path
        # L1 layer knob (ar-three-layer-spec): 0 = fp32 partials,
        # 1 = e4m3 wire (per-128 scale), 2 = nvfp4 wire (e2m1 + per-16
        # e4m3 scale), 3 = fp16 wire (near-lossless middle codec,
        # 2 B/elem). Out-of-range falls back to fp32 with a warning.
        self.wire = wire if wire in (0, 1, 2, 3) else 0
        # Two-tier wire: decode/verify messages are 5-15 KB and latency-bound, so
        # they stay uncompressed (self.wire); prefill payloads are MB-sized and
        # bandwidth-bound, where e4m3 roughly halves the traffic (ledger
        # 04_standards.md: 42 MB vs 73.5 MB per rank, 0.57x, quality gate passed).
        # Set VLLM_SM70_AR_WIRE_LARGE equal to VLLM_SM70_AR_WIRE to disable.
        _wl = os.environ.get("VLLM_SM70_AR_WIRE_LARGE", "3")
        self.wire_large = int(_wl) if _wl.isdigit() and int(_wl) in (0, 1, 2, 3) else self.wire
        self._wire_logged: str | None = None
        if self.wire != wire:
            logger.warning(
                "SM70 hier AR: wire=%r out of range {0,1,2,3}, "
                "falling back to 0 (fp32 partials)",
                wire,
            )
        self.lib = lib_path
        self._fa: int = 0
        self._ring = None  # AR-2L L1 ring (prefill valve)
        self._ring_base: int = 0
        self._base_ptr: int = 0
        self._n_max: int = 0
        # Buffer sizing: mode-2 (or any file-set boot, since the file may
        # flip to 2 at runtime) needs the prefill-sized region; otherwise
        # the decode/verify cap is enough.
        self._sized_prefill = (mode_env >= 2) or bool(file_path)
        # The env/file value is the INPUT byte cap (spec §3 eligibility);
        # the IPC allocation is the LAYOUT size for that cap (meta + 2 in
        # slots + 2 region slots + probe — the C++ buf_size op).
        self._cap_bytes = (
            _PREFILL_MAX_BYTES if self._sized_prefill else max_bytes)
        try:
            self._n_max = int(
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_n_max(self._cap_bytes))
            self._alloc_bytes = int(
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_buf_size(
                    self._cap_bytes))
            if self._sized_prefill:
                logger.info(
                    "SM70 hier AR: buffers sized to the prefill point "
                    "(cap %d B, alloc %d B/rank) — mode-2 capable",
                    self._cap_bytes, self._alloc_bytes)
            if not self._check_topology():
                return  # fail-closed (already logged)
            ptrs = self._alloc_ipc_buffers(self._alloc_bytes)
            self._base_ptr = ptrs[self.rank]  # bind early: probe failure
                                              # still frees my buffer
            self._fa = torch.ops._C_sm70_hier_ar.sm70_hier_ar_init(
                self.rank, self._cap_bytes, ptrs, 1 if lib_path else 0)
            # Host-mapped counters: polling them is plain memory access, so it
            # stays legal while the engine captures CUDA graphs.
            set_ar_stats_ptr(
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_stats_host_ptr(self._fa)
            )
            start_ar_stats_logger()
            if not self._ipc_probe(ptrs):
                self._free()
                return  # fail-closed (already logged)
            self.disabled = False
            self._init_ring(ptrs)
            logger.info(
                "SM70 hier AR: route ACTIVE (rank %d, mode env=%d, file=%r, "
                "wire=%d, lib=%r, n_max=%d elems, buffer=%d B)",
                self.rank, mode_env, file_path, self.wire, lib_path,
                self._n_max, self._alloc_bytes)
        except Exception as e:  # noqa: BLE001 - probe/alloc failure -> NCCL
            self._free()
            logger.warning(
                "SM70 hier AR: boot probe failed, FAILING CLOSED to the "
                "NCCL path: %r", e)

    # ------------------------------------------------------------------
    # Boot-time probe (fail-closed on ANY mismatch — spec §6 risk R1)
    # ------------------------------------------------------------------
    def _check_topology(self) -> bool:
        """Verify the LINK-LEVEL topology (topo_matrix: nvidia-smi topo -m,
        HW_GUIDE §3.1): exactly the 16 NVLink pairs (12 intra-quad + 4
        mirror) of _EXPECTED_P2P_PAIRS are NV-labeled AND each one is
        P2P-accessible; the other 12 (SYS) pairs are UNCONSTRAINED — the
        driver's capability query reports PCIe-P2P True on some of them
        (measured (0,4)/(0,5)/(0,6), ar_smoke4) and the route never uses
        them. So the capability query classifies NOTHING here: the link
        matrix is the classification source, the capability query is the
        16-leg health pre-filter (the transfer-level verification is the
        two-phase IPC readback below). The clique + mirror-pair table is
        the ONLY topology input to the design — boot-verified config
        (spec R4: post-rewire = re-run the matrix, re-derive the table,
        re-run gate-1; no kernel redesign).

        The decision is COLLECTIVE (no-hang on the failure path): the local
        scan feeds ONE all_reduce (MAX over a scalar int8 flag on the same
        group the probe uses) that every rank — including one whose local
        scan raised or mismatched — must reach, so a single-rank transient
        divergence (e.g. a P2P-cache subprocess failure on one rank) fails
        the WHOLE group closed in lockstep (all 8 ranks -> disabled, no
        allocation, NCCL fallback) instead of letting the other ranks
        proceed into the IPC all_gather/barrier below while the odd rank
        exits (which would hang boot in gloo)."""
        bad: list[str] = []
        local_fail = False
        ws = self.world_size
        try:
            phys = visible_ordinal_to_physical(ws)
            matrix = topo_matrix()
            # (a) LINK-LEVEL classification: the NV pair set must be the
            # 16-table exactly (a rewire = a missing/extra NV pair).
            nv: set[frozenset[int]] = set()
            for i in range(ws):
                for j in range(i + 1, ws):
                    if matrix[phys[i]][phys[j]].startswith("NV"):
                        nv.add(frozenset((i, j)))
            missing = _EXPECTED_P2P_PAIRS - nv
            extra = nv - _EXPECTED_P2P_PAIRS
            if missing or extra:
                local_fail = True
                bad.append("NV pair set mismatch: missing="
                           f"{sorted(map(sorted, missing))} extra="
                           f"{sorted(map(sorted, extra))}")
            # (b) leg health: every NV pair must be P2P-accessible. The
            # 12 SYS pairs are UNCONSTRAINED — the driver reports
            # PCIe-P2P True on some of them (measured (0,4)/(0,5)/(0,6))
            # and the route never uses them.
            for pair in sorted(_EXPECTED_P2P_PAIRS, key=sorted):
                i, j = sorted(pair)
                try:
                    if not torch.cuda.can_device_access_peer(i, j):
                        local_fail = True
                        bad.append(f"{[i, j]} NVLink pair not "
                                   "P2P-accessible")
                except Exception as e:  # noqa: BLE001
                    logger.warning("SM70 hier AR: P2P check %s failed: "
                                   "%r", [i, j], e)
                    local_fail = True
                    bad.append(f"{[i, j]} P2P check raised")
        except Exception as e:  # noqa: BLE001 - a scan crash is a local fail
            logger.warning("SM70 hier AR: topology scan failed: %r — "
                           "failing this rank's local decision closed", e)
            local_fail = True
        # ONE collective call, lockstep with the probe collectives below:
        # every rank reaches it (even on local exception above).
        flag = torch.tensor([1 if local_fail else 0], dtype=torch.int8)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
        if int(flag.item()) != 0:
            if not local_fail:
                logger.warning(
                    "SM70 hier AR: rank %d's local topology scan PASSED but "
                    "a PEER rank's scan failed (group-aggregated MAX) — "
                    "failing closed in lockstep with the group to the NCCL "
                    "path.", self.rank)
            else:
                logger.warning(
                    "SM70 hier AR: topology check failed (group-aggregated "
                    "fail; rank %d local bad-pairs: %s) — FAILING CLOSED "
                    "to the NCCL path (re-derive the clique/mirror table; "
                    "spec R4).", self.rank, bad)
            return False
        return True

    def _alloc_ipc_buffers(self, buf_bytes: int) -> list[int]:
        """House pattern (create_shared_buffer): cudaMalloc + IPC handle +
        allgather + open on every peer. Every rank allocates the SAME size
        (world-uniform here: the buffer size depends only on envs that are
        identical across ranks at boot)."""
        pointer, handle = torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(
            buf_bytes)
        handles = [None] * self.world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ptrs: list[int] = []
        for i, h in enumerate(handles):
            if i == self.rank:
                ptrs.append(pointer)
            else:
                ptrs.append(int(torch.ops._C_custom_ar.open_mem_handle(h)))
        return ptrs

    def _ipc_probe(self, ptrs: list[int]) -> bool:
        """Data-path IPC round-trip on all 16 legs (each leg probed from
        both ends across the 8 ranks): my rank id is read back through my
        3 clique legs + my mirror leg (C++ 1-thread probe kernel). A 0 on
        any rank fails the route (the route is only as good as its legs).
        Phase 1 (host writes, all ranks) -> barrier -> phase 2 (device
        reads) mirrors the house custom_all_reduce.py L388-395 ordering:
        peer-visible reads only after every peer's write is committed.

        The decision is COLLECTIVE (no-hang on the failure path — mirrors
        the F3 topology flag in ``_check_topology``): each phase's local
        work fails into a local flag instead of raising. A rank whose
        probe_write (pre-barrier) throws must still reach the barrier
        (a stray raise used to strand the other 7 ranks inside it — boot
        hang), and a rank whose probe_read/verify (post-barrier) throws
        or reads a 0 must still reach the ONE all_reduce (MAX over the
        scalar int8 flag on the same group) that every rank reaches — a
        stray local raise used to make a LOCAL disabled decision on one
        rank while the other 7 enabled the route (split-state world:
        mismatched collectives on the first allreduce). Only the
        group-aggregated result decides: a fail on ANY rank (local OR
        peer, via the MAX) disables the route on ALL 8 ranks in lockstep
        (buffers freed, NCCL fallback)."""
        n_max = self._n_max
        probe_off = 64 + 4 * n_max + 12 * n_max  # meta + in_buf + region
        probe_bases = [p + probe_off for p in ptrs]
        probe_fail = False
        try:
            torch.ops._C_sm70_hier_ar.sm70_hier_ar_probe_write(
                self._fa, self.rank)
        except Exception as e:  # noqa: BLE001 - local fail -> the flag; the
            # rank must still reach the barrier below (never skip it).
            logger.warning(
                "SM70 hier AR: probe write failed on rank %d: %r — failing "
                "this rank's local probe decision closed (still reaching "
                "the barrier + the group flag)", self.rank, e)
            probe_fail = True
        dist.barrier(group=self.group)
        if not probe_fail:
            try:
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_probe_read(
                    self._fa, self.rank, probe_bases)
                ok = torch.ops._C_sm70_hier_ar.sm70_hier_ar_probe_ok(self._fa)
                if not ok:
                    logger.warning(
                        "SM70 hier AR: local IPC probe failed on rank %d — "
                        "failing this rank's local probe decision closed "
                        "(the group-aggregated flag below decides)",
                        self.rank)
                    probe_fail = True
            except Exception as e:  # noqa: BLE001 - local fail -> the flag;
                # the rank must still reach the group all_reduce below.
                logger.warning(
                    "SM70 hier AR: probe read/verify failed on rank %d: %r "
                    "— failing this rank's local probe decision closed "
                    "(still reaching the group flag)", self.rank, e)
                probe_fail = True
        # ONE collective call, lockstep with the F3 topology flag: every
        # rank reaches it (even one that threw in either phase above).
        agreed_probe_fail = torch.tensor(
            [1 if probe_fail else 0], dtype=torch.int8)
        dist.all_reduce(agreed_probe_fail, op=dist.ReduceOp.MAX,
                        group=self.group)
        if int(agreed_probe_fail.item()) != 0:
            if not probe_fail:
                logger.warning(
                    "SM70 hier AR: rank %d's local probe PASSED but a PEER "
                    "rank's probe failed (group-aggregated MAX) — failing "
                    "closed in lockstep with the group to the NCCL path.",
                    self.rank)
            else:
                logger.warning(
                    "SM70 hier AR: IPC probe failed on at least one of the "
                    "16 NVLink legs (rank %d local fail; group-aggregated "
                    "flag) — FAILING CLOSED to the NCCL path.", self.rank)
            return False
        return True

    # ------------------------------------------------------------------
    # Per-launch mode (R2-T1b control file) + dispatch
    # ------------------------------------------------------------------
    def _mode(self) -> int:
        if self.file:
            m = _read_mode_file(self.file, default=self.mode_env)
            return self.mode_env if m <= 0 else m
        return self.mode_env

    def should_use(self, inp: torch.Tensor) -> bool:
        """Eligibility (spec §3; ALL must hold, else fall-through = today's
        path): mode on (control file wins per launch, garbage -> the
        static env value) · fp16 · contiguous · C++-safe shape guards
        (the shadow path must NEVER raise where the house path would not)
        · decode/verify size cap (the
        prefill flat-AR leg is NOT hijacted).

        The cheap guards (dtype/contiguity/size) run BEFORE the per-launch
        mode read: the control file is opened only for an ELIGIBLE launch
        (spec §3 "re-read per eligible launch") — zero file I/O on the
        default-off path and on every ineligible tensor (the 96/
        decode-step hot path)."""
        if self.disabled:
            return False
        if inp.dtype is not torch.float16:
            return False
        if not inp.is_contiguous():
            return False
        n = inp.numel()
        if _AR_REJECT_LOG and n > self._n_max:
            logger.info("SM70AR reject n=%d bytes=%d n_max=%d", n,
                        n * inp.element_size(), self._n_max)
        if n > self._n_max:
            return False
        if n % 128 != 0:  # wire group size (the C++ op would raise; guard)
            if _AR_REJECT_LOG:
                logger.info("SM70AR reject-align n=%d bytes=%d", n,
                            n * inp.element_size())
            return False
        # Per-launch mode — opens the control file ONLY now, after the
        # cheap guards (garbage -> the static env value, R2-T1b pattern).
        mode = self._mode()
        if mode < 1:
            return False
        # Mode 1: decode/verify cap only. Mode 2: + the prefill point, but
        # only if the buffers were sized for it at boot (the n_max guard
        # above already closes the non-sized case: a non-sized buffer has
        # n_max for the decode cap only).
        # The ring route (prefill) needs prefill-sized payloads to reach
        # all_reduce() at all; without this the cap rejects them here and the
        # collective silently falls through to NCCL.
        ring_armed = self._ring is not None and _route_enabled()
        if _AR_ADAPTIVE and not ring_armed:
            route, _wire = _choose_route_wire(
                self._phase(), n * inp.element_size(), self.max_bytes)
            if route != "hier":
                _trace_take("SKIP-ADAPTIVE", n * inp.element_size(), self.max_bytes)
                return False
        cap = _PREFILL_MAX_BYTES if (
            (mode >= 2 or ring_armed) and self._sized_prefill
        ) else self.max_bytes
        if _AR_REJECT_LOG and n * inp.element_size() > cap:
            logger.info("SM70AR reject-cap n=%d bytes=%d cap=%d", n,
                        n * inp.element_size(), cap)
        if n * inp.element_size() > cap:
            _trace_take("SKIP-CAP", n * inp.element_size(), cap)  # TEMP-AB
            return False
        _trace_take("TAKE", n * inp.element_size(), cap)  # TEMP-AB
        return True

    def _init_ring(self, ptrs: list[int]) -> None:
        """AR-2L L1 ring: prefill-only, behind an explicit env valve.

        Fail-open to the hier route on any problem: the ring is an experiment
        until the ruler and the content gate say otherwise.
        """
        import os as _os

        if _os.environ.get("VLLM_SM70_AR_RING_PREFILL", "0") != "1":
            return
        if self.world_size != 8:
            raise RuntimeError(
                f"ring route requested with world_size={self.world_size}; "
                "this kernel is defined for 8 ranks (rule R1: no silent skip)")
        try:
            from vllm.distributed.device_communicators.sm70_ar_ring import (
                RingAR,
                find_nv_ring,
            )

            rows = topo_matrix()
            ring = find_nv_ring(rows, self.world_size)
            if ring is None:
                raise RuntimeError(
                    "ring route requested but the box has no NVLink-only "
                    "Hamiltonian cycle (rule R1: no silent fallback)")
            pos = ring.index(self.rank)
            prev_rank = ring[(pos - 1) % self.world_size]
            from vllm.distributed.device_communicators.sm70_ar_ring import (
                WIRE as _RING_WIRE,
            )
            from vllm.distributed.device_communicators.sm70_ar_ring import (
                region_bytes as _ring_region_bytes,
            )

            need = _ring_region_bytes(self.world_size, 256, self._cap_bytes,
                                      _RING_WIRE)
            rptrs = self._alloc_ipc_buffers(need)
            self._ring_base = rptrs[self.rank]
            self._ring = RingAR(
                wire=self.wire_large,  # prefill route: compressed wire
                rank=self.rank,
                world=self.world_size,
                prev_rank=prev_rank,
                base_ptr=rptrs[self.rank],
                prev_ptr=rptrs[prev_rank],
                cap_bytes=self._cap_bytes,
                device=int(self.device.index if self.device.index is not None
                           else torch.cuda.current_device()),
            )
            logger.info(
                "SM70 AR ring: ACTIVE (rank %d, order=%s, prev=%d, cap=%d B)",
                self.rank, ring, prev_rank, self._cap_bytes)
        except Exception as e:  # rule R1: no fallback, fail the boot loudly
            logger.error("SM70 AR ring: init failed (%r) — refusing to start "
                         "with a silent fallback (rule R1)", e)
            raise

    def _phase(self) -> str:
        """Which engine route issued this collective.

        Decode and verify run inside a captured CUDA graph
        (cudagraph_runtime_mode == FULL); prefill runs eagerly (NONE/PIECEWISE).
        That is the route the engine actually took, unlike a size threshold.
        Capture itself is reported separately so a pre-capture call never picks
        the prefill wire by accident.
        """
        try:
            # No is_current_stream_capturing() probe: a capture is always a
            # decode-shaped graph here (prefill runs eagerly), so FULL means the
            # decode wire either way, and the probe would cost a CUDA query on
            # every one of the ~96 collectives per step.
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
        except Exception:  # noqa: BLE001 - never change behaviour on a probe miss
            return "unknown"
        mode = getattr(ctx, "cudagraph_runtime_mode", None)
        if getattr(mode, "name", "") == "FULL":
            return "decode"
        bd = getattr(ctx, "batch_descriptor", None)
        if bd is not None and getattr(bd, "uniform", False):
            nreq = getattr(bd, "num_reqs", None)
            if nreq is None or getattr(bd, "num_tokens", 0) == nreq:
                return "decode"
        return "prefill"

    def _wire_for(self, inp: torch.Tensor) -> int:
        if _AR_ADAPTIVE:
            wire = _choose_route_wire(
                self._phase(), inp.numel() * inp.element_size(),
                self.max_bytes)[1]
            if getattr(self, "_adaptive_logged", None) != wire:
                self._adaptive_logged = wire
                logger.info("SM70AR adaptive: bytes=%d wire=%d",
                            inp.numel() * inp.element_size(), wire)
            return wire
        wire = self.wire
        if self.wire_large != self.wire:
            phase = self._phase()
            # Guard on size as well: the phase probe reports "prefill" during
            # warmup/profiling runs whose messages are decode-sized (5 KB), and
            # quantizing those would cost precision with no bandwidth to save.
            if phase == "prefill" and inp.numel() * inp.element_size() > self.max_bytes:
                wire = self.wire_large
            if phase != self._wire_logged:
                self._wire_logged = phase
                logger.info(
                    "SM70AR wire: phase=%s bytes=%d wire=%d (small=%d large=%d)",
                    phase, inp.numel() * inp.element_size(), wire,
                    self.wire, self.wire_large,
                )
        return wire

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if _AR_BYPASS:
            return inp
        if _AR_TRACE:
            _ar_trace_note()
        out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        ring = self._ring
        if (ring is not None and nbytes > self.max_bytes
                and _route_enabled() and ring.eligible(inp)):
            # Prefill-sized payload: the ring is the bandwidth route (one
            # collective per step, handshakes fully amortised) and it runs the
            # compressed wire. Decode-sized payloads stay on the clique kernel
            # below -- they are latency-bound and must not be quantised.
            # Payloads the ring cannot represent (below its 1 MB floor) take the
            # clique kernel instead; the route is logged either way.
            _route_note("prefill", nbytes, "ring", ring.wire)
            _RING_STATS["calls"] += 1
            _RING_STATS["bytes"] += int(nbytes)
            ring.all_reduce(inp, out)
            return out
        wire = self._wire_for(inp)
        _route_note("decode" if nbytes <= self.max_bytes else "prefill",
                    nbytes, "clique", wire)
        torch.ops._C_sm70_hier_ar.sm70_hier_ar_reduce(self._fa, inp, out, wire)
        return out

    def close(self) -> None:
        self._free()
        self.disabled = True

    def _free(self) -> None:
        if self._fa:
            try:
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_free(self._fa)
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass
            self._fa = 0
        if getattr(self, "_ring_base", 0):
            try:
                torch.ops._C_custom_ar.free_shared_buffer(self._ring_base)
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass
            self._ring_base = 0
        self._ring = None
        if self._base_ptr:
            try:
                torch.ops._C_custom_ar.free_shared_buffer(self._base_ptr)
            except Exception:  # noqa: BLE001 - teardown best-effort, same
                # as the fa free above: a poisoned CUDA context (e.g. the
                # async IMA from the probe kernel) must not crash the
                # worker — the route is already disabled and the buffer is
                # reclaimed at process exit
                pass
            self._base_ptr = 0


def should_build(
    unique_name: str,
    world_size: int,
    device: torch.device | None,
) -> bool:
    """Static hook gate (spec §3): SM70 platform · tp group · world 8 ·
    env gate on (mode > 0) or control file set. Read at init only — with
    everything at defaults this returns False and the module is never
    imported by the hot path (rule 11670)."""
    if world_size != 8:
        return False
    if unique_name.split(":")[0] != "tp":
        return False
    if not (envs.VLLM_SM70_AR_HIER > 0 or envs.VLLM_SM70_AR_HIER_FILE):
        return False
    if not current_platform.is_cuda():
        return False
    try:
        # interface signature: is_device_capability(cap, device_id: int)
        # — the route's OWN device (every rank checks its own GPU).
        device_id = (
            device.index
            if device is not None and device.index is not None
            else 0
        )
        if not current_platform.is_device_capability((7, 0), device_id):
            return False
    except Exception:  # noqa: BLE001 - unknown capability -> stay off
        return False
    return True
