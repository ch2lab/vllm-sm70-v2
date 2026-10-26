"""Offline unit test for ple_shm.py: daemon-style allocation + engine-style
mapping of one named /dev/shm PLE file, cross-verify shared pages."""

import os

import torch


def main() -> None:
    # cudaHostRegister needs a live CUDA context in this process.
    _ = torch.zeros(1, device="cuda:0")

    from vllm.model_executor.model_loader.weight_cache.ple_shm import (
        allocate_ple_shm,
        close_ple_shm,
        map_ple_shm,
        mark_ple_shm,
    )

    shape, dtype = (4096, 80), torch.uint8

    # --- daemon side: allocate ---
    t = allocate_ple_shm("test_a", shape, dtype)
    path = getattr(t, "_wcm_ple_shm", None)
    assert path is not None and os.path.isfile(path), "allocate must expose the shm path"
    assert t.is_pinned(), "allocated PLE must be pinned (cudaHostRegister)"
    assert t.shape == torch.Size(shape) and t.dtype == dtype
    t[7] = torch.arange(80, dtype=torch.uint8)

    # --- engine side: map the same file ---
    t2 = map_ple_shm(path, shape, dtype)
    assert t2.is_pinned(), "mapped PLE must be pinned (UVA would otherwise copy)"
    assert t2.shape == torch.Size(shape) and t2.dtype == dtype
    assert torch.equal(t[7], torch.arange(80, dtype=torch.uint8)), "must read daemon's bytes"

    # Cross-process visibility: write through the engine view, read back on the
    # daemon view.
    t2[11] = torch.full((), 0xAB, dtype=torch.uint8)
    assert int(t[11, 0]) == 0xAB, "daemon view must see the engine's write"

    # --- marker ---
    mark_ple_shm(t, path)
    assert t._wcm_ple_shm == path

    # --- missing file raises FileNotFoundError ---
    try:
        map_ple_shm("/dev/shm/vllm_wcm_ple_definitely_missing_42", (4, 80), torch.uint8)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass

    # --- Parameter wrapping regression: a re-wrap loses the tensor-level
    #     attributes; the daemon re-mark pattern must move the handle with
    #     the path, or exit cleanup (close via the wrapped object) silently
    #     skips the file's unlink (T4b finding: 8 leaked codes files). ---
    from torch import nn

    wrapped = nn.Parameter(t, requires_grad=False)
    assert getattr(wrapped, "_wcm_ple_shm_handle", None) is None, (
        "wrapping must not inherit the handle (that was the bug)"
    )
    mark_ple_shm(wrapped, path)
    wrapped._wcm_ple_shm_handle = t._wcm_ple_shm_handle
    t._wcm_ple_shm_handle = None

    # --- close: engine first (no unlink), daemon last (unlink, via the
    #     wrapped object that exit cleanup actually holds) ---
    close_ple_shm(t2, unlink=False)
    assert os.path.exists(path), "unlink=False must keep the file for the daemon"
    close_ple_shm(wrapped, unlink=True)
    assert not os.path.exists(path), "close via wrapped param must unlink"

    print("ple_shm OK")


if __name__ == "__main__":
    main()
