#!/usr/bin/env python3
"""Reproduce/verify the SM70 flat 2-hop all-reduce at one payload shape.

Runs in its own process per shape so an illegal memory access cannot poison
other measurements. Compares against NCCL bitwise and reports latency.

Run: python -m torch.distributed.run --nproc_per_node=4 \
        sm70_tests/repro_flat_ar_shape.py --rows 64
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

HIDDEN = 5120
AR_PER_STEP = 128


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--max-bytes", type=int, default=1 << 20)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.sm70_flat_ar import Sm70FlatAR

    flat = Sm70FlatAR(
        group=cpu_group,
        device=torch.device(f"cuda:{rank}"),
        max_bytes=args.max_bytes,
        s_max=8,
    )
    rows = args.rows
    x = torch.randn(rows, HIDDEN, dtype=torch.float16, device=f"cuda:{rank}")
    ref = x.clone()
    dist.all_reduce(ref)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"rows={rows} n={x.numel()} bytes={x.numel() * 2} "
              f"cap={args.max_bytes} should_use={flat.should_use(x)}")
    if not flat.should_use(x):
        print(f"[rank {rank}] shapes outside the flat route; nothing to test")
        dist.destroy_process_group()
        return

    got = flat.all_reduce(x)
    torch.cuda.synchronize()
    same = torch.equal(got, ref)
    print(f"[rank {rank}] bitwise_equal={same} "
          f"max_abs={(got.float() - ref.float()).abs().max().item():.3e}")
    # Production runs the AR inside FULL decode CUDA graphs: capture it, then
    # replay. This is the path the plain eager bench never exercised.
    g = torch.cuda.CUDAGraph()
    static_in = x.clone()
    torch.cuda.synchronize()
    dist.barrier()
    try:
        with torch.cuda.graph(g):
            static_out = flat.all_reduce(static_in)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        replay_ref = static_in.clone()
        dist.all_reduce(replay_ref)
        torch.cuda.synchronize()
        print(f"[rank {rank}] graph_replay bitwise_equal="
              f"{torch.equal(static_out, replay_ref)} max_abs="
              f"{(static_out.float() - replay_ref.float()).abs().max().item():.3e}")
    except Exception as exc:  # noqa: BLE001 - repro script
        print(f"[rank {rank}] graph capture failed: {type(exc).__name__}: {exc}")

    nccl_us = bench(lambda: dist.all_reduce(x)) * 1000
    flat_us = bench(lambda: flat.all_reduce(x)) * 1000
    if rank == 0:
        print(f"nccl {nccl_us:.1f} us | flat {flat_us:.1f} us | "
              f"speedup {nccl_us / flat_us:.2f}x | "
              f"flat {flat_us * AR_PER_STEP / 1000:.2f} ms/step")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
