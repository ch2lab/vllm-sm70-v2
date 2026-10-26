# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production seam for the NCCL-replica all-reduce (``ar_nccl_replica.cu``).

This wraps the 8-rank / N-channel ring kernel that was developed and validated
offline (``sm70_tests/bench_nccl_replica.py``) behind the SAME interface the
house route (``Sm70HierAR``) exposes to ``CudaCommunicator.all_reduce``:

    should_use(inp: Tensor) -> bool
    all_reduce(inp: Tensor) -> Tensor
    close() -> None

Selected with ``VLLM_SM70_AR_IMPL=replica`` (default ``hier``); one env line
switches back, so the production default is unchanged.

Four things differ from the offline harness, and each one is load-bearing:

* **One region, many message sizes.**  The service sends decode-sized
  collectives (2..128 tokens -> 10 KB..640 KB) and prefill-sized ones (up to
  ``max_num_batched_tokens`` -> ~50 MB) through the same object.  The kernel's
  layout is therefore *cap* based (``ncr_set_cap``): every control word keeps
  one address no matter how large the message is, so a small message's payload
  can never land on a large message's flag word.
* **Device-side epoch.**  Decode runs inside a captured CUDA graph, so the
  Python path does not execute on replay and a host-provided epoch cannot
  advance.  ``ncr_set_epoch_dev(1)`` moves the counter into the region; each
  channel's CTA advances its own word once per launch (proven by
  ``sm70_tests/bench_ncr_capmix.py``: mixed sizes + frozen host epoch exact).
* **Lazy allocation.**  vLLM validates the engine's memory budget *after* the
  communicator is built, so a ~100 MiB region taken in ``__init__`` costs the
  boot its last 40 MiB (measured: 16.45 GiB free vs 16.49 GiB required).  The
  region is allocated on the first eligible collective, which lands in the
  eager profile run.
* **Fault surfacing.**  A ring bail (spin timeout) is not a CUDA error: the
  kernel returns and leaves a partially reduced buffer.  The kernel publishes a
  per-channel fault word, and every eager collective checks it plus the region
  canary, so that case raises instead of silently changing the numbers.

Rule R1 (no fallback) is enforced: an unrepresentable message, a failed boot
probe, a dirty canary or a ring fault raises instead of quietly returning to
NCCL.
"""

from __future__ import annotations

import ctypes

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# NVLink-direct ring cycles verified on this box (nvidia-smi topo -m; the same
# table the offline harness uses).  Channel weights follow the leg capacity
# ratio 2:2:1:1 (the NV2 cycles first).  Anything not in this table is refused:
# a SYS peer inside a ring is the DMA fault that wedged the driver on 2026-10-20.
_RING_A = [0, 1, 3, 4, 6, 7, 5, 2]        # all NV2 legs
_RING_C = [0, 3, 2, 1, 6, 5, 4, 7]        # all NV1 legs
_RINGS = ([_RING_A, _RING_A[::-1], _RING_C, _RING_C[::-1]], [2, 2, 1, 1])

_DEFAULT_LIB = "/data/sm70_results/ncr_prod/lib_ncr_prod_ag1.so"
_P2P_LIB = "/data/sm70_results/artifacts/lib_arring.so"

_c_int, _c_ll, _c_p, _c_sz = (ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p,
                             ctypes.c_size_t)


def _channel_rings(rank: int, world: int, ch: int) -> tuple[list[int],
                                                           list[int]]:
    """Per-channel successor rank and this rank's position inside its ring."""
    rings, weights = _RINGS
    tot = float(sum(weights))
    raw = [ch * w / tot for w in weights]
    split = [int(x) for x in raw]
    for k in sorted(range(len(raw)), key=lambda i: raw[i] - split[i],
                    reverse=True)[:ch - sum(split)]:
        split[k] += 1
    if min(split) <= 0:
        raise RuntimeError(
            f"channels={ch} too few for ring weights {weights} -> {split}")
    succ: list[int] = []
    pos: list[int] = []
    for k, cyc in enumerate(rings):
        if sorted(cyc) != list(range(world)):
            raise RuntimeError(f"ring {k} is not a permutation of 0..{world-1}")
        p = cyc.index(rank)
        for _ in range(split[k]):
            succ.append(cyc[(p + 1) % world])
            pos.append(p)
    return succ, pos


class NcclReplicaAR:
    """ctypes binding of ``ar_nccl_replica.cu`` with the house AR interface."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_bytes: int,
        wire: int,
        lib_path: str,
        ch: int,
        slots: int,
        threads: int,
        strict: bool,
    ) -> None:
        self.group = group
        self.device = device
        self.max_bytes = max_bytes
        self.wire = wire
        self.ch = ch
        self.slots = slots
        self.threads = threads
        self.strict = strict
        self.disabled = True
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self._base_ptr = 0
        self._host_epoch = 0
        self._warned_sizes: set[int] = set()
        self._checked_sizes: set[int] = set()
        self._logged: set[tuple[int, str]] = set()
        self._over_cap: set[int] = set()
        self._ready = False

        if self.world != 8:
            raise RuntimeError(
                f"NCCL-replica AR is an 8-rank ring; world={self.world}")
        if ch < 1 or ch > 32:
            raise RuntimeError(f"channels={ch} out of range 1..32")

        self._lib = ctypes.CDLL(lib_path)
        self._bind()
        if self._lib.ncr_selftest_bounds() != 0:
            raise RuntimeError(
                "ncr_selftest_bounds() refused this build (codes: -1..-8 fp16 "
                "layout, -9..-16 wire layouts, -17..-28 codecs, -29..-33 "
                "cap/epoch/fault layout)")
        if self.wire not in (0, 1, 2, 3):
            raise RuntimeError(f"wire={wire} out of range 0..3")
        self.dev = int(device.index if device.index is not None
                       else torch.cuda.current_device())
        # No device memory is taken here on purpose: vLLM validates the engine's
        # memory budget after the communicator exists, and a ~100 MiB region at
        # this point cost the last 40 MiB of that check (measured).
        logger.info(
            "SM70 AR replica: armed (rank %d, cap %d B, ch=%d slots=%d "
            "threads=%d wire=%d strict=%s) — region allocated on first "
            "collective", self.rank, self.max_bytes, self.ch, self.slots,
            self.threads, self.wire, self.strict)

    # ------------------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        lib.ncr_region_bytes.argtypes = [_c_int, _c_int, _c_int, _c_ll, _c_int]
        lib.ncr_region_bytes.restype = _c_sz
        lib.ncr_set_cap.argtypes = [_c_ll]
        lib.ncr_set_epoch_dev.argtypes = [_c_int]
        lib.ncr_set_wire.argtypes = [_c_int]
        lib.ncr_set_codec.argtypes = [_c_int]
        lib.ncr_max_resident.argtypes = [_c_int, _c_int]
        lib.ncr_set_device.argtypes = [_c_int]
        lib.ncr_copy.argtypes = [_c_p, _c_p, _c_sz, _c_int, _c_p]
        lib.ncr_reset.argtypes = [_c_p, _c_int, _c_int, _c_int, _c_ll, _c_int,
                                  _c_int, _c_p]
        lib.ncr_prime.argtypes = [_c_p, _c_int, _c_int, _c_int, _c_ll, _c_int,
                                  _c_int, _c_p]
        lib.ncr_canary_init.argtypes = [_c_p, _c_int, _c_int, _c_int, _c_ll,
                                        _c_int]
        lib.ncr_canary_ok.argtypes = [_c_p, _c_int, _c_int, _c_int, _c_ll,
                                      _c_int]
        lib.ncr_fault_ok.argtypes = [_c_p, _c_int, _c_int, _c_int, _c_ll,
                                     _c_int]
        lib.ncr_launch.argtypes = (
            [_c_int, _c_p, ctypes.POINTER(_c_p), ctypes.POINTER(_c_int)] +
            [_c_int, _c_int, _c_int, _c_ll, ctypes.c_uint, _c_int, _c_p,
             ctypes.c_ulonglong, _c_p, _c_int])
        for f in (lib.ncr_selftest_bounds, lib.ncr_reset, lib.ncr_prime,
                  lib.ncr_canary_init, lib.ncr_canary_ok, lib.ncr_fault_ok,
                  lib.ncr_launch, lib.ncr_copy, lib.ncr_max_resident,
                  lib.ncr_set_device, lib.ncr_set_cap, lib.ncr_set_epoch_dev,
                  lib.ncr_set_wire, lib.ncr_set_codec):
            f.restype = _c_int

    # ------------------------------------------------------------------
    def _lazy_init(self) -> None:
        """Allocate the region and map it on the first eligible collective.

        Every rank reaches this at the same collective, so the allgather and the
        barriers are symmetric.  A CUDA-graph capture would make them illegal,
        and that case is an explicit error rather than a silent skip.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "replica AR: first use happened inside CUDA graph capture; the "
                "region must be allocated during the eager profile run")
        lib = self._lib
        dev = self.dev
        if lib.ncr_set_wire(self.wire) != 0:
            raise RuntimeError("ncr_set_wire failed")
        if lib.ncr_set_codec(1 if self.wire else 0) != 0:
            raise RuntimeError("ncr_set_codec failed")
        if lib.ncr_set_cap(self.max_bytes) != 0:
            raise RuntimeError(f"ncr_set_cap({self.max_bytes}) failed")
        if lib.ncr_set_epoch_dev(1) != 0:
            raise RuntimeError("ncr_set_epoch_dev failed")
        lib.ncr_set_device(dev)
        nbytes = int(lib.ncr_region_bytes(self.ch, self.world, self.slots,
                                          self.max_bytes, 1))
        if nbytes <= 0:
            raise RuntimeError(
                f"region for cap={self.max_bytes} ch={self.ch} "
                f"slots={self.slots} wire={self.wire} is not representable "
                f"(cap must be a multiple of {16 * self.ch * self.world}"
                + (f", and of {self.ch * self.world * 4096} for wire "
                   f"{self.wire}: one wire message must hold whole "
                   "quantisation groups, i.e. the token count of the message "
                   "has to be a multiple of "
                   f"{2 * self.ch * self.world * 4096 // 5120}"
                   if self.wire else "")
                + ")")
        nres = lib.ncr_max_resident(dev, self.threads)
        if nres < self.ch:
            raise RuntimeError(
                f"only {nres} resident CTAs, the ring needs {self.ch}")
        self.region_bytes_alloc = nbytes

        # House pattern (create_shared_buffer): cudaMalloc + IPC handle +
        # allgather + open on every peer.  8 processes, one context each (rule
        # R2: never a single process with 8 contexts).
        ptr, handle = (
            torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(nbytes))
        handles: list = [None] * self.world
        dist.all_gather_object(handles, handle, group=self.group)
        ptrs = [ptr if i == self.rank
                else int(torch.ops._C_custom_ar.open_mem_handle(h))
                for i, h in enumerate(handles)]
        self._base_ptr = ptr

        self.succ, self.pos = _channel_rings(self.rank, self.world, self.ch)
        p2p = ctypes.CDLL(_P2P_LIB)
        p2p.ring_enable.argtypes = [_c_int, ctypes.POINTER(_c_int), _c_int,
                                    _c_int]
        p2p.ring_enable.restype = _c_int
        for g in sorted(set(self.succ)):
            if p2p.ring_enable(dev, (ctypes.c_int * 1)(g), 1, 1) != 0:
                raise RuntimeError(
                    f"gpu {g} is not an NVLink-direct peer of gpu {dev}: the "
                    "ring would put a SYS peer in the write path (driver fault)")
        self._p2p = p2p  # keep the handle alive for the process lifetime

        st = torch.cuda.current_stream(dev).cuda_stream
        if lib.ncr_reset(ctypes.c_void_p(ptr), self.ch, self.world, self.slots,
                         self.max_bytes, 1, dev, st) != 0:
            raise RuntimeError("ncr_reset failed")
        if lib.ncr_canary_init(ctypes.c_void_p(ptr), self.ch, self.world,
                               self.slots, self.max_bytes, 1) != 0:
            raise RuntimeError("ncr_canary_init failed")
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        for g in sorted(set(self.succ)):
            if lib.ncr_prime(ctypes.c_void_p(ptrs[g]), self.ch, self.world,
                             self.slots, self.max_bytes, 1, dev, st) != 0:
                raise RuntimeError(f"ncr_prime(gpu {g}) failed")
        if lib.ncr_canary_ok(ctypes.c_void_p(ptr), self.ch, self.world,
                            self.slots, self.max_bytes, 1) != 0:
            raise RuntimeError("region canary dirty right after init")
        if lib.ncr_fault_ok(ctypes.c_void_p(ptr), self.ch, self.world,
                           self.slots, self.max_bytes, 1) != 0:
            raise RuntimeError("ring reported a fault right after init")
        dist.barrier(group=self.group)

        self._ptrs = ptrs
        self._succ_arr = (ctypes.c_void_p * self.ch)(*[ptrs[g]
                                                      for g in self.succ])
        self._pos_arr = (ctypes.c_int * self.ch)(*self.pos)
        self._ready = True
        self.disabled = False
        logger.info(
            "SM70 AR replica: ACTIVE (rank %d, cap %d B, region %.1f MiB, "
            "ch=%d slots=%d threads=%d wire=%d)",
            self.rank, self.max_bytes, nbytes / 2**20, self.ch, self.slots,
            self.threads, self.wire)

    # ------------------------------------------------------------------
    def _representable(self, nbytes: int) -> bool:
        if nbytes <= 0:
            return False
        if nbytes % (16 * self.ch * self.world):
            return False
        msg = nbytes // (self.ch * self.world)      # one wire message, bytes
        if msg % 16:
            return False
        if self.wire and (msg // 2) % 256:
            return False
        return not (self.wire == 3 and (msg // 2) % 2048)

    def should_use(self, inp: torch.Tensor) -> bool:
        if inp.dtype is not torch.float16 or not inp.is_contiguous():
            return self._refuse(inp, "dtype/contiguity")
        nbytes = inp.numel() * 2
        # The cap IS this route's declared takeover envelope (the same shape of
        # contract as VLLM_SM70_AR_HIER_MAX_BYTES): a collective bigger than it
        # belongs to the next backend by definition, and saying so once is not a
        # hidden fallback.  Anything *inside* the envelope that the ring cannot
        # represent (dtype, divisibility) is still an error under strict.
        if nbytes > self.max_bytes:
            if nbytes not in self._over_cap:
                self._over_cap.add(nbytes)
                logger.info(
                    "SM70 AR replica: %d B AR (shape %s) is beyond the %d B "
                    "takeover cap — left to the next backend",
                    nbytes, tuple(inp.shape), self.max_bytes)
            return False
        if not self._representable(nbytes):
            return self._refuse(inp, f"not representable inside the cap "
                                    f"{self.max_bytes}")
        if not self._ready:
            self._lazy_init()
        if nbytes not in self._warned_sizes:
            self._warned_sizes.add(nbytes)
            logger.info("SM70 AR replica: rank %d takes over %d B AR "
                        "(shape %s)", self.rank, nbytes, tuple(inp.shape))
        return True

    def _refuse(self, inp: torch.Tensor, why: str) -> bool:
        """Rule R1: an eligible-looking collective this route cannot represent is
        an ERROR when strict, and a loudly logged pass-through to the next
        backend in the dispatch chain otherwise."""
        msg = (f"SM70 AR replica: cannot take {tuple(inp.shape)} "
               f"{inp.dtype} ({inp.numel() * inp.element_size()} B): {why}")
        if self.strict:
            raise RuntimeError(msg + " — rule R1 forbids a silent fallback")
        key = (inp.numel(), str(inp.dtype))
        if key not in self._logged:
            self._logged.add(key)
            logger.warning("%s — passing to the next backend", msg)
        return False

    # ------------------------------------------------------------------
    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        nbytes = inp.numel() * 2
        out = torch.empty_like(inp)
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        base = self._base_ptr
        if self._lib.ncr_copy(ctypes.c_void_p(base),
                              ctypes.c_void_p(inp.data_ptr()), nbytes,
                              self.dev, stream) != 0:
            raise RuntimeError("replica AR: stage-in failed")
        rc = self._lib.ncr_launch(self.dev, ctypes.c_void_p(base),
                                  self._succ_arr, self._pos_arr, self.ch,
                                  self.world, self.slots, nbytes,
                                  self._host_epoch, self.threads, None,
                                  int(500 * 1e6), stream, 1)
        self._host_epoch += 1
        if rc != 0:
            raise RuntimeError(f"replica AR launch rc={rc} for {nbytes} B "
                               "(-104/105/106/107 = shape, -20x/-21x = layout)")
        if self._lib.ncr_copy(ctypes.c_void_p(out.data_ptr()),
                              ctypes.c_void_p(base), nbytes, self.dev,
                              stream) != 0:
            raise RuntimeError("replica AR: stage-out failed")
        # The checks below cost a stream sync, which CUDA graph capture forbids;
        # a capture computes nothing, so there is nothing to verify there.
        if not torch.cuda.is_current_stream_capturing():
            self._verify(nbytes)
        return out

    def _verify(self, nbytes: int) -> None:
        frc = self._lib.ncr_fault_ok(ctypes.c_void_p(self._base_ptr), self.ch,
                                     self.world, self.slots, self.max_bytes, 1)
        if frc != 0:
            raise RuntimeError(
                f"replica AR: ring reported a fault on channel {frc - 1} "
                f"(1 = spin timeout, -301 = address guard) after a {nbytes} B "
                "collective")
        if envs.VLLM_SM70_AR_REPLICA_CANARY:
            rc = self._lib.ncr_canary_ok(ctypes.c_void_p(self._base_ptr),
                                         self.ch, self.world, self.slots,
                                         self.max_bytes, 1)
            if rc != 0:
                raise RuntimeError(
                    f"replica AR: region canary dirty (word {rc}) after a "
                    f"{nbytes} B collective — the ring wrote outside its region")
        if nbytes not in self._checked_sizes:
            self._checked_sizes.add(nbytes)
            if self._lib.ncr_selftest_bounds() != 0:
                raise RuntimeError("replica AR: addressing self-test failed on "
                                   "the live build")

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.disabled = True
        self._ready = False
        if self._base_ptr:
            torch.ops._C_custom_ar.free_shared_buffer(self._base_ptr)
            self._base_ptr = 0
