#!/usr/bin/env python3
"""Latency of the TP4 all-reduce at the decode payloads.

The conc8 budget attributes ~0.5 ms per extra verify row to the 128 per-step
all-reduces (2 per layer x 64 layers).  This measures the two implementations
we ship (NCCL and the SM70 flat 2-hop route) at exactly the payloads used by
decode, so the AR share of the row cost stops being an estimate.

Run: python -m torch.distributed.run --nproc_per_node=4 sm70_tests/bench_ar_latency.py
"""
import os
import time

import torch
import torch.distributed as dist

HIDDEN = 5120
ITERS = 200
AR_PER_STEP = 128          # 2 row-parallel ARs per layer x 64 layers


def bench(fn) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / ITERS


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    cpu_group = dist.new_group(backend="gloo")

    flat = None
    try:
        from vllm.distributed.device_communicators.sm70_flat_ar import Sm70FlatAR

        flat = Sm70FlatAR(
            group=cpu_group,
            device=torch.device(f"cuda:{rank}"),
            max_bytes=1 << 22,
            s_max=8,
        )
    except Exception as exc:  # noqa: BLE001 - measurement script
        if rank == 0:
            print(f"flat AR unavailable: {type(exc).__name__}: {exc}")

    if rank == 0:
        print(f"world={world} hidden={HIDDEN} fp16, {AR_PER_STEP} AR/step")
        print(f"{'rows':>5} {'nccl us':>9} {'flat us':>9} {'nccl ms/step':>13} "
              f"{'flat ms/step':>13}")
    for rows in (1, 8, 16, 24, 32):
        x = torch.randn(rows, HIDDEN, dtype=torch.float16, device=f"cuda:{rank}")
        nccl_us = bench(lambda: dist.all_reduce(x)) * 1000
        flat_us = float("nan")
        if flat is not None and flat.should_use(x):
            flat.copy_ = None
            flat_us = bench(lambda: flat.all_reduce(x)) * 1000
        if rank == 0:
            print(f"{rows:5d} {nccl_us:9.1f} {flat_us:9.1f} "
                  f"{nccl_us * AR_PER_STEP / 1000:13.1f} "
                  f"{flat_us * AR_PER_STEP / 1000:13.1f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
