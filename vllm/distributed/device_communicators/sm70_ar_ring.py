"""AR-2L · L1 ring allreduce adapter (production side).

Valve: ``VLLM_SM70_AR_RING_PREFILL=1``.  Only large messages take this path;
decode keeps the existing clique+mirror route (its latency is ~32 us/hop here,
which is the wrong shape for batch-1).

L1 carries opaque fp16 bytes.  The e4m3 wire is a separate L2 step and is NOT
enabled here: on a ring every element is forwarded N-1 times, so the codec cost
scales with the hop count (measured: +18% net, not 1.98x).
"""

from __future__ import annotations

import ctypes
import logging

import torch

logger = logging.getLogger(__name__)

LIB = "/data/sm70_results/artifacts/lib_arring_ar.so"
LIB_WIRE = "/data/sm70_results/artifacts/lib_arring_wire.so"
# L2 wire: 0 = fp16 (verified), 1 = e4m3, 2 = nvfp4 (both gated by the
# boundary test first; see sm70_tests/bench_wire_boundary.py)
WIRE = int(__import__("os").environ.get("VLLM_SM70_AR_RING_WIRE", "0"))
# Below this size the ring loses to the low-latency clique+mirror route
# (32 us/hop); prefill collectives are 7-42 MB, decode ones are ~5-20 KB.
MIN_BYTES = 1 << 20
NV_LINKS = ("NV2", "NV1")


def _align256(x: int) -> int:
    return (x + 255) & ~255


def meta_bytes(npieces: int, nchunk: int) -> int:
    return _align256(4 * (npieces * nchunk + 2))


def region_bytes(npieces: int, nchunk: int, bytes_: int, wire: int = 0) -> int:
    """Wire regions add payload (bytes/2) and scales (bytes/32: the max of the
    e4m3 per-128 and nvfp4 per-16 needs)."""
    base = meta_bytes(npieces, nchunk) + bytes_
    return base + (bytes_ // 2 + bytes_ // 32 if wire else 0)


def find_nv_ring(rows: dict[int, list[str]], n: int) -> list[int] | None:
    """A Hamiltonian cycle over NVLink-only edges (NV2 preferred)."""

    def search(allow_nv1: bool) -> list[int] | None:
        ok = ("NV2",) if not allow_nv1 else NV_LINKS
        adj = {i: [j for j in range(n) if j != i and rows[i][j] in ok] for i in range(n)}
        path, used = [0], {0}

        def dfs() -> bool:
            if len(path) == n:
                return 0 in adj[path[-1]]
            for j in adj[path[-1]]:
                if j in used:
                    continue
                path.append(j)
                used.add(j)
                if dfs():
                    return True
                path.pop()
                used.discard(j)
            return False

        return path if dfs() else None

    return search(False) or search(True)


class RingARError(RuntimeError):
    """Raised instead of falling back: rule R1 forbids silent degradation."""


class RingAR:
    """One instance per rank.  base_ptr / prev_ptr are IPC-mapped regions."""

    def __init__(
        self,
        rank: int,
        world: int,
        prev_rank: int,
        base_ptr: int,
        prev_ptr: int,
        cap_bytes: int,
        device: int,
        nchunk: int = 256,
        nblocks: int = 128,
        threads: int = 256,
        wire: int | None = None,
    ) -> None:
        self.rank = rank
        self.world = world
        self.prev = prev_rank
        self.base = base_ptr
        self.prev_base = prev_ptr
        self.cap = cap_bytes
        self.device = device
        self.nchunk = nchunk
        self.nblocks = nblocks
        self.threads = threads
        self.data_off = meta_bytes(world, nchunk)
        self.epoch = 0
        self.wire = WIRE if wire is None else wire
        self.lib = ctypes.CDLL(LIB_WIRE if self.wire else LIB)
        self._bind()

    def _bind(self) -> None:
        """Bind only what the selected library actually exports."""
        L = self.lib
        ci, cp, cv, cs, cu = (ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_void_p)
        if self.wire:
            self._set_device = L.wire_set_device
            self._copy = L.wire_copy
            L.wire_mark.argtypes = [cv, ci, ci, ci, ci, ci, cp]
            L.wire_launch.argtypes = [ci, cv, cv, ci, ci, ci, ci, ci, ci, ci,
                                      ci, cp]
            L.wire_mark.restype = ci
            L.wire_launch.restype = ci
        else:
            self._set_device = L.ring_ar_set_device
            self._copy = L.ring_ar_copy
            L.ring_ar_mark.argtypes = [cv, ci, ci, ci, cp]
            L.ring_ar_launch_stream.argtypes = [ci, cv, cv, ci, ci, ci, ci, ci,
                                                ci, ci, cp]
            L.ring_ar_mark.restype = ci
            L.ring_ar_launch_stream.restype = ci
        self._set_device.argtypes = [ci]
        self._set_device.restype = ci
        self._copy.argtypes = [cv, cv, cs, cp]
        self._copy.restype = ci

    def eligible(self, inp: torch.Tensor) -> bool:
        """Pure predicate.  The call site decides; no hidden fallback."""
        if inp.dtype != torch.float16:
            return False
        if not inp.is_contiguous():
            return False
        nbytes = inp.numel() * 2
        # The old predicate demanded the payload fit the constructor's fixed
        # nchunk, which no single value satisfies across prefill sizes (the
        # library also needs chunk_elems % group == 0). Ask the picker instead.
        return (nbytes >= MIN_BYTES and nbytes <= self.cap
                and self._pick_nchunk(nbytes) is not None)

    def _pick_nchunk(self, nbytes: int) -> int | None:
        """Largest nchunk <= self.nchunk whose layout the library accepts.

        Layout::make() rejects a shape unless piece_elems % nchunk == 0 AND
        chunk_elems % group == 0 (group = 128 for the e4m3 wire, 16 otherwise),
        so a fixed nchunk cannot serve every prefill payload: 32 MB with
        nchunk=256 gives chunk_elems=8000, and 8000 % 128 = 64 -> the mark call
        fails. Search downwards instead of hard-coding.
        """
        nelem = nbytes // 2
        if nelem % self.world:
            return None
        piece = nelem // self.world
        group = 128 if self.wire == 1 else 16
        for nc in (self.nchunk, 256, 128, 64, 32, 16, 8, 4, 2, 1):
            if nc > self.nchunk or piece % nc:
                continue
            if (piece // nc) % group:
                continue
            return nc
        return None

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Ring allreduce.  Any problem raises (rule R1: no fallback)."""
        if not self.eligible(inp):
            raise RingARError(
                f"shape not representable for the ring: numel={inp.numel()} "
                f"dtype={inp.dtype} contig={inp.is_contiguous()} cap={self.cap}")
        if out.dtype != torch.float16 or not out.is_contiguous():
            raise RingARError("output must be contiguous fp16")
        nbytes = inp.numel() * 2
        nchunk = self._pick_nchunk(nbytes)
        if nchunk is None:
            raise RingARError(
                f"no representable nchunk for {nbytes} B at wire={self.wire}")
        data_off = meta_bytes(self.world, nchunk)
        stream = ctypes.c_void_p(
            torch.cuda.current_stream(self.device).cuda_stream
        )
        L = self.lib
        self._set_device(self.device)
        my = ctypes.c_void_p(self.base + data_off)
        prev = ctypes.c_void_p(self.prev_base + data_off)
        self.epoch += 1
        # Wrap into the meta ring: it holds `nchunk` slots shared by `world`
        # ranks, so the slot base must be modulo nchunk/world. Without this the
        # ring fails its own mark validation from collective #33 onwards
        # (measured 2026-10-22: the warmup alone issues more than that, so the
        # route never survived a real boot).
        base = (self.epoch % max(1, nchunk // self.world)) * self.world
        if self._copy(my, ctypes.c_void_p(inp.data_ptr()), nbytes, stream):
            raise RingARError("staging copy failed")
        # publish only now: peers must never see meta=1 before the payload lands
        if self.wire:
            if L.wire_mark(ctypes.c_void_p(self.base), self.world, nchunk,
                           nbytes, self.wire, base, stream):
                logger.error(
                    "SM70AR ring mark rejected: world=%d nchunk=%d nbytes=%d "
                    "wire=%d chunk_elems=%d data_off=%d cap=%d",
                    self.world, nchunk, nbytes, self.wire,
                    (nbytes // 2 // self.world) // nchunk, data_off, self.cap)
                raise RingARError("wire_mark failed")
            if L.wire_launch(self.device, ctypes.c_void_p(self.base),
                             ctypes.c_void_p(self.prev_base), nbytes, self.world,
                             nchunk, self.nblocks, self.threads, self.rank,
                             base, self.wire, stream):
                raise RingARError("wire_launch failed")
        else:
            if L.ring_ar_mark(ctypes.c_void_p(self.base), self.world, nchunk,
                              base, stream):
                raise RingARError("ring_ar_mark failed")
            if L.ring_ar_launch_stream(
                self.device, ctypes.c_void_p(self.base),
                ctypes.c_void_p(self.prev_base), nbytes, self.world, nchunk,
                self.nblocks, self.threads, self.rank, base, stream,
            ):
                raise RingARError("ring_ar_launch_stream failed")
        if self._copy(ctypes.c_void_p(out.data_ptr()), my, nbytes, stream):
            raise RingARError("result copy failed")
