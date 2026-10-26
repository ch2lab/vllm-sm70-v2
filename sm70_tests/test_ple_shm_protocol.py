"""Offline protocol probe: TensorEntry shm export -> pickle round-trip ->
engine-side map, verifying the shared path end to end without a live daemon."""

import pickle

import torch


def main() -> None:
    _ = torch.zeros(1, device="cuda:0")  # CUDA context for register

    from vllm.model_executor.model_loader.weight_cache.ple_shm import (
        allocate_ple_shm,
        close_ple_shm,
        map_ple_shm,
    )
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        TensorEntry,
    )

    shape, dtype = (8192, 128), torch.uint8

    # --- daemon side: export ---
    codes = allocate_ple_shm("probe_codes", shape, dtype)
    entry = TensorEntry.from_tensor(codes, "param")
    assert entry.shm_name == codes._wcm_ple_shm, "shm entry must carry the path"
    assert entry.cpu_tensor is None and entry.ipc_args is None
    assert entry.shape == shape and entry.dtype == "torch.uint8"

    # rebuild must refuse (shared entries are mapped, never copied)
    try:
        entry.rebuild(0)
        raise AssertionError("rebuild must refuse shm entries")
    except NotImplementedError:
        pass

    # --- socket transport: pickle round-trip ---
    wire = pickle.loads(pickle.dumps(entry))
    assert wire.shm_name == entry.shm_name and wire.shape == shape

    # --- engine side: map (same dtype decode as ipc_loader._apply_entries) ---
    mapped = map_ple_shm(wire.shm_name, wire.shape, getattr(torch, "uint8"))
    assert mapped.is_pinned()
    codes[3] = torch.arange(128, dtype=torch.uint8)
    assert torch.equal(mapped[3], codes[3]), "engine must read daemon's pages"
    mapped[5] = torch.full((), 0x77, dtype=torch.uint8)
    assert int(codes[5, 0]) == 0x77, "daemon must see engine writes"

    close_ple_shm(mapped, unlink=False)
    close_ple_shm(codes, unlink=True)
    print("ple_shm protocol probe OK")


if __name__ == "__main__":
    main()
