"""Named shared-memory PLE storage shared between the weight-cache daemon and engines.

The weight-cache daemon allocates its PLE weight (NVFP4 codes + scales) into a
named file under /dev/shm, maps it with MAP_SHARED and pins the mapping with
``cudaHostRegister``. Engines open the *same file* and map + register their
own view, so the UVA helper (``get_cuda_view_from_cpu_tensor``) reads the
daemon's pages directly instead of silently allocating a private pinned copy.

Precedent: the in-tree Engram path in
``vllm/models/deepseek_v41/common/engram.py`` (named /dev/shm file + mmap +
cudaHostRegister + ``is_pinned()`` check).

Lifecycle contract:
- The daemon's file lives as long as the daemon; only the daemon unlinks it
  (via ``close_ple_shm(..., unlink=True)`` on its exit path).
- Engines never unlink; they register/unregister their own mapping.
- Unlinking a file that is still mapped keeps the pages valid for every open
  mapping (POSIX semantics), so the daemon's unlink on exit cannot corrupt a
  live engine view.

mlock / RLIMIT notes: ``cudaHostRegister`` charges the process memlock limit;
the per-rank PLE (~3.6 GB) is well inside the 15.4 GB per-process limit of the
SM70 deployment. Registration failure is a hard error: a non-pinned PLE would
be swappable and would silently destroy decode latency.
"""

import contextlib
import mmap
import os
import uuid

import numpy as np
import torch

SHM_DIR = "/dev/shm"
SHM_PREFIX = "vllm_wcm_ple_"

# cudaHostRegister requires a page-aligned size; round the file and the
# registration up to this granularity (a multiple of every x86_64 page size).
_PAGE = 64 * 1024


class _PleShmHandle:
    """Owns the file descriptor, mapping and CUDA registration of one view.

    Stored on the tensor as ``_wcm_ple_shm_handle`` so its lifetime equals the
    tensor's. The daemon and each engine hold distinct handles on the same
    file.
    """

    def __init__(
        self,
        path: str,
        fd: int,
        mapping: mmap.mmap,
        pointer: int,
        reg_bytes: int,
        owns_file: bool,
    ) -> None:
        self.path = path
        self.fd = fd
        self.mapping = mapping
        self.pointer = pointer
        self.reg_bytes = reg_bytes
        self.owns_file = owns_file
        self.closed = False

    def close(self, unlink: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        result = torch.cuda.cudart().cudaHostUnregister(self.pointer)
        if result.value != 0:
            # A failed unregister must not hide the backing store from other
            # mappings: leave the file intact and surface the error.
            raise RuntimeError(f"cudaHostUnregister failed: {result}")
        os.close(self.fd)
        if unlink and self.owns_file:
            os.unlink(self.path)
        # The tensor still exports the mmap buffer through numpy, so an
        # explicit close can raise BufferError until the tensor is released;
        # let GC finish the unmap then (same as the Engram finalizer path).
        with contextlib.suppress(BufferError):
            self.mapping.close()


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _page_rounded(num_bytes: int) -> int:
    return (num_bytes + _PAGE - 1) // _PAGE * _PAGE


def _register_and_view(
    fd: int,
    num_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, mmap.mmap, int, int]:
    """mmap the file fd, pin the mapping, return (tensor, mapping, ptr, reg)."""
    reg_bytes = _page_rounded(num_bytes)
    mapping = mmap.mmap(fd, reg_bytes, flags=mmap.MAP_SHARED)
    try:
        owner = np.frombuffer(mapping, dtype=np.uint8)
        pointer = owner.ctypes.data
        result = torch.cuda.cudart().cudaHostRegister(pointer, reg_bytes, 0)
        if result.value != 0:
            raise RuntimeError(f"cudaHostRegister failed: {result}")
        tensor = torch.from_numpy(owner[:num_bytes]).view(*shape).view(dtype)
        if not tensor.is_pinned():
            # The UVA helper otherwise allocates a private pinned copy of the
            # PLE (~3.6 GB per rank) and the sharing silently does not happen.
            raise RuntimeError(
                "cudaHostRegister did not pin the PLE shared mapping; "
                "refusing to serve a non-pinned PLE"
            )
    except BaseException:
        mapping.close()
        raise
    return tensor, mapping, pointer, reg_bytes


def _open_fd(path: str, create: bool) -> int:
    """Create or open the shm file; raises FileNotFoundError when missing."""
    if create:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    return os.open(path, os.O_RDWR)


def _mark(tensor: torch.Tensor, path: str, handle: "_PleShmHandle") -> None:
    tensor._wcm_ple_shm = path  # type: ignore[attr-defined]
    tensor._wcm_ple_shm_handle = handle  # type: ignore[attr-defined]


def allocate_ple_shm(
    label: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Allocate a fresh named shm PLE file and return a pinned tensor view.

    The file name embeds ``label`` (e.g. ``"daemon_r0_<uuid8>"``) plus a random
    suffix so concurrent ranks never collide. The tensor carries:
      - ``_wcm_ple_shm``: absolute file path (the value engines use to map)
      - ``_wcm_ple_shm_handle``: owns fd/mapping/registration of this view
    Raises on any failure (file collision, register failure, non-pinned).
    """
    num_bytes = _numel(shape) * torch.tensor([], dtype=dtype).element_size()
    if num_bytes == 0:
        raise ValueError(f"PLE shm shape {shape} has zero elements")
    path = f"{SHM_DIR}/{SHM_PREFIX}{label}_{uuid.uuid4().hex[:8]}"
    fd = _open_fd(path, create=True)
    try:
        os.ftruncate(fd, _page_rounded(num_bytes))
        tensor, mapping, pointer, reg_bytes = _register_and_view(
            fd, num_bytes, shape, dtype
        )
    except BaseException:
        os.close(fd)
        os.unlink(path)
        raise
    handle = _PleShmHandle(
        path=path,
        fd=fd,
        mapping=mapping,
        pointer=pointer,
        reg_bytes=reg_bytes,
        owns_file=True,
    )
    _mark(tensor, path, handle)
    return tensor


def map_ple_shm(
    path: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Map an existing daemon-created shm PLE file and return a pinned view.

    Raises FileNotFoundError when the path is missing (the caller — the
    engine's entry application — translates that into its disk fallback).
    """
    num_bytes = _numel(shape) * torch.tensor([], dtype=dtype).element_size()
    if num_bytes == 0:
        raise ValueError(f"PLE shm shape {shape} has zero elements")
    fd = _open_fd(path, create=False)
    try:
        size = os.fstat(fd).st_size
        if size < num_bytes:
            raise RuntimeError(
                f"PLE shm file {path} is {size} bytes, expected >= {num_bytes}"
            )
        tensor, mapping, pointer, reg_bytes = _register_and_view(
            fd, num_bytes, shape, dtype
        )
    except BaseException:
        os.close(fd)
        raise
    handle = _PleShmHandle(
        path=path,
        fd=fd,
        mapping=mapping,
        pointer=pointer,
        reg_bytes=reg_bytes,
        owns_file=False,
    )
    _mark(tensor, path, handle)
    return tensor


def mark_ple_shm(tensor: torch.Tensor, path: str) -> None:
    """Stamp a tensor with its shm path (protocol + reload guards read this)."""
    tensor._wcm_ple_shm = path  # type: ignore[attr-defined]


def close_ple_shm(tensor: torch.Tensor, unlink: bool = False) -> None:
    """Unregister and unmap the tensor's shm view.

    ``unlink`` (daemon exit path only) removes the file; engine-side closes
    must not unlink the daemon's file. No-op when the tensor has no handle.
    """
    handle: _PleShmHandle | None = getattr(
        tensor, "_wcm_ple_shm_handle", None
    )
    if handle is None or handle.closed:
        tensor._wcm_ple_shm = ""  # type: ignore[attr-defined]
        tensor._wcm_ple_shm_handle = None  # type: ignore[attr-defined]
        return
    handle.close(unlink=unlink)
    tensor._wcm_ple_shm = ""  # type: ignore[attr-defined]
    tensor._wcm_ple_shm_handle = None  # type: ignore[attr-defined]
