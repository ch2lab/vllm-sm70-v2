#!/usr/bin/env python3
"""Offline check of the production Sm70FlatAR class (bit-exact, any TP).

  # TP8 on devices 0-7
  /data/vllm-dev/bin/python sm70_tests/test_flat_ar_class.py --n 2560 --iters 100
  # TP4 on devices 4-7 (fully NVLink connected -> single-hop route)
  /data/vllm-dev/bin/python sm70_tests/test_flat_ar_class.py --world 4 \
      --devices 4,5,6,7 --n 2560 --iters 100

The reference is the fp32 sum of the gathered inputs in rank order, i.e. the
exact reduction order the kernel promises, so ``bit-exact`` must hold for every
world size.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, init_file: str, ns: list[int], iters: int, results,
           world: int, devices: list[int]) -> None:
    sys.path.insert(0, "/data/src/vllm-v2")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "0"
    dev = devices[rank]
    torch.cuda.set_device(dev)
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world)
    from vllm.distributed.device_communicators.sm70_flat_ar import Sm70FlatAR

    ar = Sm70FlatAR(group=None, device=f"cuda:{dev}", max_bytes=1 << 20,
                    s_max=4)
    if ar.disabled:
        raise RuntimeError(f"rank {rank}: flat AR failed closed (topology)")
    out_rows = []
    for n in ns:
        torch.manual_seed(1234 + rank)
        inp = torch.randn(n, device=f"cuda:{dev}", dtype=torch.float16)
        gathered = [None] * world
        dist.all_gather_object(gathered, inp.cpu())
        ref = torch.zeros(n, dtype=torch.float32, device=f"cuda:{dev}")
        for g in gathered:
            ref += g.to(ref.device).float()
        ref = ref.half()

        assert ar.should_use(inp), f"should_use False for n={n}"
        # warm-up (the first calls ramp)
        for _ in range(200):
            out = ar.all_reduce(inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = ar.all_reduce(inp)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        exact = bool(torch.equal(ref, out))

        # CUDA graph path: this is what the engine actually replays.
        static_in = inp.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = ar.all_reduce(static_in)
        for _ in range(50):
            g.replay()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        us_graph = (time.perf_counter() - t1) / iters * 1e6
        exact_graph = bool(torch.equal(ref, static_out))

        d = (ref.float() - out.float()).abs().max().item()
        out_rows.append((n, us, exact and exact_graph, d))
        if rank == 0:
            print(f"n={n:<7d} eager {us:7.2f} us/call  graph {us_graph:7.2f} "
                  f"us/call  bit-exact={exact and exact_graph}  "
                  f"max_abs={d:.5f}")
        dist.barrier()
    if rank == 0:
        results["rows"] = out_rows
    dist.barrier()
    ar.close()
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[2560, 40960])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--devices", type=str, default="",
                    help="comma separated physical GPU per rank "
                         "(default 0..world-1)")
    a = ap.parse_args()
    devices = ([int(d) for d in a.devices.split(",") if d] if a.devices
               else list(range(a.world)))
    if len(devices) != a.world:
        ap.error(f"--devices needs {a.world} entries, got {len(devices)}")
    init_file = f"/data/sm70_results/flatclass_{os.getpid()}"
    pathlib.Path(init_file).touch()
    mgr = mp.Manager()
    results = mgr.dict()
    mp.spawn(worker, args=(init_file, a.n, a.iters, results, a.world, devices),
             nprocs=a.world, join=True)
    try:
        os.unlink(init_file)
    except FileNotFoundError:
        pass
    rows = results.get("rows") or []
    return 0 if all(r[2] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
