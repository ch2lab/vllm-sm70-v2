#!/usr/bin/env python3
"""Gate-1 offline microbench — SM70 topology-aware hierarchical AR shadow
route vs the current NCCL path (round-3 AR, spec §4).

Contract: .superpowers/sdd/2026-10-13-sm70-round3-ar/ar-direction-spec.md
(Run in a quiet window. Window discipline G0 is enforced below and the
script REFUSES TO START when any precondition is not met.)

Run (single entry; spawns 8 worker processes, one per V100):

    source /data/src/sm70_env.sh
    /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/bench_ar_hier_gate1.py \
        [--verify-m 64] [--rounds 300] [--stress 8700] [--no-skew] \
        [--out /data/sm70_results/ar_hier_gate1_YYYYMMDD]

FIRST REAL GPU LAUNCH ON THIS BOX: must be `--smoke` (R3 lesson). `--smoke`
is SMOKE (non-gating): M=1 only, rounds capped at 10, no stress, no skew —
a first-launch sanity config whose verdict is labeled "SMOKE (non-gating)"
and is NEVER a gate-1 verdict (re-run without --smoke for the real window).

Arms (both driven through the SAME wired seam, `CudaCommunicator.all_reduce`
— the exact dispatch the service will use; spec §4 "Arms"):
  A = current NCCL path (pynccl; at 5 KB the service runs RING_LL — matches
      the trace kernel name; CUMEM off per HW_GUIDE §3.5).
  B = shadow hierarchical route (graph-captured). Built in mode 2 so the
      IPC buffers are prefill-sized and every shape is route-eligible; the
      leg/protocol work at M=1/M=verify is identical to mode 1 (only the
      size cap differs).

Window discipline (G0, HARD):
  * journal silent >= 60 s at window start (`journalctl -u sm70-serve`);
  * service log idle (== journal silence);
  * <= 1.5 GB/GPU peak working set per arm (measured per rank: torch
    allocator peak + the route's raw IPC buffer);
  * ZERO service API calls (this script makes none; no HTTP client is
    imported; the serve is observed only via journalctl/dmesg on disk);
  * interleaved arms in one process (A,B per round, barrier-separated);
  * same-day paired (A and B in this single launch; verdict stamps the day).
  * probe-process safety (HW_GUIDE §3.5): NCCL_CUMEM_ENABLE=0 in every
    worker; on a hang the parent waits 90 s for self-abort, then TERM
    one by one (NO kill -9) and marks the window INVALID (no auto re-run).

ARM ORDERING RULE (deadlock avoidance, documented for the reader): NCCL (A)
and the hierarchical protocol (B) are two independent collective "clocks".
If ranks interleave A/B replays in different orders, a rank's B kernel
spins on a peer that is stuck inside A's NCCL wait (and vice versa) ->
global hang. Therefore every arm round is bracketed by a gloo barrier:
all 8 ranks finish the A round before any rank starts B. The barrier is
CPU-side and sits OUTSIDE the timed window (device events e_in/e_out are
recorded inside the captured graph). The barrier-to-replay launch gap is
the measured "input-ready skew" — the same quantity the service sees from
per-layer compute completion.

Pre-registered gates (spec §4 — fixed before any run):
  G0 window   (HARD) — the preconditions above.
  G1 parity   (HARD) — B bitwise-equal across all 8 ranks at every shape;
                      B finite; max_abs(B, FP64-ref) <= max_abs(A, FP64-ref).
  G2 graph-join (HARD) — B per-call median <= 0.90 x A at M=1 AND M=verify,
                      OR B's coordination term (B median - pure-wire lower
                      bound from the spec §1.3 leg table) <= 1.0 us with
                      collective end after all inputs ready B < A.
  G3 prefill  (INFO) — M=8192 B/A ratio recorded; expected B >= 1.4 x A
                      (hierarchical loses -> CONFIRMS the routing rule; a
                      surprise win is reported, not auto-adopted).
  G4 stress   (HARD) — 8,700-collective pure CUDA-graph replay on B
                      completes (no hang/spin/replay-skew stop) + re-run on
                      identical inputs -> byte-identical output.
  G5 Xid      (HARD) — no NEW Xid on any of the 8 GPUs (dmesg diff).
  G6 VRAM     (HARD) — peak torch-allocator delta B vs A per rank <= 0
                      AND the route's raw cudaMalloc IPC buffer (outside
                      the torch allocator; spec: "expected 0 by
                      construction; still measured") <= 2.2 MiB/rank for
                      the production mode-1 class; the mode-2/file-set
                      bench sizing (~335.5 MiB/rank, the prefill point) is
                      allowed ONLY inside the 1.5 GB working-set cap,
                      which counts the raw route bytes per rank.

Outputs: <out>/rank0.json + <out>/gate1.json (machine verdict) + a human
summary on stdout. Exit code: 0 = all HARD gates PASS, 2 = HARD fail(s),
1 = window never valid (G0 refused or hang).

O1 (verify-batch M): derived, never guessed — block_size from the live
draft model's dflash_config (scanned under /data/models/*-DFlash2*/) x
max-num-seqs from the serve run script; override with --verify-m.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SERVE_UNIT = "sm70-serve"
RESULTS_DIR = Path(os.environ.get("SM70_RESULTS_DIR", "/data/sm70_results"))
WORKING_SET_CAP_BYTES = int(1.5 * 1024 * 1024 * 1024)  # spec: <=1.5 GB/GPU/arm
HIDDEN = 2560  # production AR tensor width (spec §1.2)
WORLD = 8
# G6 raw-IPC-buffer bound for the PRODUCTION mode-1 class (pre-registered;
# spec: shadow buffers "expected 0 by construction; still measured"). Cited
# mode-1 cap sizing (sm70_hier_ar.py: alloc = 64 + 16*n_max + 8 with
# n_max = ceil128(cap/2); production VLLM_SM70_AR_HIER_MAX_BYTES default
# 262144 B -> 2,097,224 B ~= 2.0 MiB/rank): the mode-1 allocation must stay
# <= 2.2 MiB. The mode-2/file-set bench sizing (~335.5 MiB/rank, the
# prefill point) is NOT bounded here: it is allowed only inside the 1.5 GB
# working-set cap (which counts the raw route bytes per rank — the
# working_set_ok_per_rank check in the G6 gather below).
G6_ROUTE_IPC_MODE1_MAX_BYTES = int(2.2 * 1024 * 1024)  # 2,306,867 B

# ---------------------------------------------------------------------------
# G0 window discipline — parent side, BEFORE any GPU work
# ---------------------------------------------------------------------------


def _journal_last_age_s() -> float | None:
    """Age (s) of the LAST serve journal line; None when unverifiable.

    `--output=short-iso` line shape:
        2026-10-13T12:09:05+08:00 aiserver5 systemd[1]: ...
    (ISO-8601 with offset, host, unit[pid]: message)."""
    try:
        out = subprocess.run(
            ["journalctl", "-u", SERVE_UNIT, "-n", "1", "--output=short-iso"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if not out:
            return None
        line = out.splitlines()[-1]
        m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S*)", line)
        if not m:
            return None
        from datetime import datetime
        ts = datetime.fromisoformat(m.group(1))
        if ts.tzinfo is None:  # host-local journal (no offset)
            age = time.time() - time.mktime(ts.timetuple())
        else:
            age = time.time() - ts.timestamp()
        return age
    except Exception:  # noqa: BLE001 - unverifiable journal -> fail closed
        return None


def _dmesg_xids() -> list[str]:
    try:
        out = subprocess.run(
            ["dmesg", "--level", "error"], capture_output=True, text=True,
            timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    return [ln.strip() for ln in out.splitlines()
            if re.search(r"\bxid\b", ln, re.IGNORECASE)]


def check_window_g0() -> dict:
    """G0 preconditions; refuses the window (SystemExit 1) when unmet."""
    age = _journal_last_age_s()
    if age is None:
        raise SystemExit(
            "G0 FAIL (refuse): serve journal unverifiable "
            f"(`journalctl -u {SERVE_UNIT}`) — need >= 60 s of journal "
            "silence BEFORE the window starts (spec §4 window discipline).")
    if age < 60.0:
        raise SystemExit(
            f"G0 FAIL (refuse): serve journal last line {age:.0f} s ago — "
            "need >= 60 s of silence (service log idle). Wait, then re-run.")
    window = {
        "window_started_unix": time.time(),
        "window_day": time.strftime("%Y-%m-%d", time.localtime()),
        "journal_silence_s": age,
        "xids_before": _dmesg_xids(),
        "same_day_paired": True,   # A and B run in this single launch
        "zero_service_api_calls": True,  # structural: no client code here
    }
    print(f"[G0] window OK: journal silent {age:.0f}s, day "
          f"{window['window_day']}, "
          f"{len(window['xids_before'])} pre-existing Xid lines", flush=True)
    return window


# ---------------------------------------------------------------------------
# O1: verify-batch M (derived, never guessed — spec §4 O1)
# ---------------------------------------------------------------------------


def derive_verify_m() -> tuple[int, str]:
    """M = block_size (draft dflash_config) x max-num-seqs (serve script)."""
    import os
    block_size = None
    for p in sorted(Path("/data/models").glob("*-DFlash2*/config.json")):
        try:
            cfg = json.loads(Path(p).read_text())
        except Exception:  # noqa: BLE001
            continue
        dflash = cfg.get("dflash_config") or {}
        if isinstance(dflash, dict) and "block_size" in dflash:
            block_size = int(dflash["block_size"])
            break
    if block_size is None:
        raise SystemExit(
            "O1 FAIL: no draft-model dflash_config.block_size under "
            "/data/models/*-DFlash2*/config.json — refusing to guess the "
            "verify-batch shape (spec O1: do not guess).")
    max_seqs = 8
    try:
        m = re.search(r"--max-num-seqs\s+(\d+)",
                      Path("/root/run/run_sm70_serve.sh").read_text())
        if m:
            max_seqs = int(m.group(1))
    except OSError:
        pass
    return block_size * max_seqs, (
        f"dflash_config.block_size={block_size} x max-num-seqs={max_seqs}")


# ---------------------------------------------------------------------------
# Worker (one per rank). Collective steps are lockstep across the 8 workers;
# arm rounds are barrier-bracketed (see the ARM ORDERING RULE in the header).
# ---------------------------------------------------------------------------


def worker(rank: int, port: int, args: dict) -> None:
    import os
    from datetime import timedelta
    import numpy as np
    import torch
    import torch.distributed as dist

    os.environ["NCCL_CUMEM_ENABLE"] = "0"  # HW_GUIDE §3.5
    # live-service parity: the service boots with this set (flashinfer
    # cubin 0.6.18.post1 vs wheel 0.7.0); CudaCommunicator imports
    # flashinfer_all_reduce unconditionally at construction
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=WORLD,
        init_method=f"tcp://127.0.0.1:{port}",
        # no dist.FOREVER in torch; the bench self-terminates (1500 s cap
        # + 90 s self-abort), so a 24 h timeout is "forever" in practice
        timeout=timedelta(hours=24),
    )
    group = dist.new_group(backend="gloo")
    # all_gather_object requires object_list pre-sized to the group size
    gw = dist.get_world_size(group)

    # Both arms through the EXACT service seam (spec §4 "Arms"). A: all
    # gates at default -> the hook is a None-check fall-through to pynccl.
    # B: shadow-route gate on (mode 2 = prefill-sized buffers). Env is read
    # at CudaCommunicator init (vllm.envs reads os.environ live).
    import vllm  # noqa: F401  (loads _C_stable_libtorch via the platform)
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    def build_comm() -> CudaCommunicator:
        return CudaCommunicator(
            group, device=device, unique_name="tp:0",
            global_ranks=list(range(WORLD)), global_world_size=WORLD)

    comm_a = build_comm()
    assert comm_a.sm70_hier_ar is None, "arm A must not build the shadow route"
    os.environ["VLLM_SM70_AR_HIER"] = "2"
    os.environ["VLLM_SM70_AR_HIER_MAX_BYTES"] = str(64 * HIDDEN * 2)
    comm_b = build_comm()
    route_b = comm_b.sm70_hier_ar
    route_ok = route_b is not None and not getattr(route_b, "disabled", True)
    dist.barrier(group)
    if rank == 0:
        print(f"[worker] arm-B shadow route: "
              f"{'ACTIVE' if route_ok else 'FAILED CLOSED -> window invalid'}",
              flush=True)
    if not route_ok:
        out = Path(args["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "rank0.json").write_text(json.dumps(
            {"error": "B route failed its boot probe (fail-closed) — "
                      "window INVALID; check the per-rank boot log"}))
        dist.destroy_process_group()
        return
    if rank == 0:
        print(f"[worker] route buffer {route_b._alloc_bytes} B/rank, "
              f"n_max={route_b._n_max}", flush=True)

    def make_inputs(m: int, seed: int = 1234) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed + rank)
        t = torch.randn(m, HIDDEN, generator=g, dtype=torch.float32)
        return (t / 8.0).to(device, dtype=torch.float16)

    results: dict = {"rank": rank}

    # ------------------------------------------------------------------
    # G1 numerical (eager; fixed per-rank seed; ref = FP64 sum on rank 0).
    # All ranks: A call (NCCL world) -> B call (hierarchical world), in the
    # same order on every rank -> no cross-arm deadlock (each collective
    # completes before any rank leaves it).
    # ------------------------------------------------------------------
    g1: dict = {}
    for m in args["shapes"]:
        inp = make_inputs(m)
        out_a = comm_a.all_reduce(inp)
        out_b = comm_b.all_reduce(inp)
        torch.cuda.synchronize()
        bytes_in = (inp.cpu().contiguous().view(torch.uint8).numpy()
                    .tobytes())
        bytes_a = (out_a.cpu().contiguous().view(torch.uint8).numpy()
                   .tobytes())
        bytes_b = (out_b.cpu().contiguous().view(torch.uint8).numpy()
                   .tobytes())
        inps, outas, outbs = [None] * gw, [None] * gw, [None] * gw
        dist.all_gather_object(inps, bytes_in, group=group)
        dist.all_gather_object(outas, bytes_a, group=group)
        dist.all_gather_object(outbs, bytes_b, group=group)
        if rank == 0:
            ref = None
            for b in inps:
                x = np.frombuffer(b, dtype=np.float16).astype(np.float64)
                ref = x if ref is None else ref + x
            ref32 = ref.astype(np.float32)

            def err_of(bs):
                e = np.stack([
                    np.abs(np.frombuffer(b, dtype=np.float16)
                           .astype(np.float32) - ref32)
                    for b in bs])
                return float(e.max()), float(e.mean()), e

            max_a, mean_a, _ = err_of(outas)
            max_b, mean_b, eb = err_of(outbs)
            b_bits = [b for b in outbs]
            g1[m] = {
                "max_abs_a": max_a, "mean_abs_a": mean_a,
                "max_abs_b": max_b, "mean_abs_b": mean_b,
                "b_bitwise_equal_across_ranks": all(
                    x == b_bits[0] for x in b_bits),
                "b_finite": bool(np.isfinite(eb).all()),
            }
        dist.barrier(group)
    results["g1"] = g1 if rank == 0 else None

    # ------------------------------------------------------------------
    # G2 graph-join timing: [compute kernel -> AR] captured per arm (own
    # graph pool => no cross-graph aliasing). The AR sub-span is measured
    # DEVICE-side: two 1-thread globaltimer() kernel nodes bracket the AR
    # (events recorded inside capture are unreadable from the host on
    # this stack — cudaErrorInvalidValue, measured on ar_smoke6c; kernel
    # nodes have no such limit). Interleaved A,B per round with barriers
    # (ARM ORDERING RULE).
    # ------------------------------------------------------------------
    dummy = torch.zeros(64, 64, device=device, dtype=torch.float16)

    def capture_graph(comm, m: int) -> dict:
        inp = make_inputs(m, seed=99)
        # In-graph AR sub-span timer slots (device-side globaltimer ns;
        # see the section note on why not CUDA events).
        ts_in = torch.zeros(1, dtype=torch.int64, device=device)
        ts_out = torch.zeros(1, dtype=torch.int64, device=device)

        def body():
            _ = dummy.mul(1.0)  # the compute-side of the join
            torch.ops._C_sm70_hier_ar.sm70_hier_ar_timer(ts_in)
            out = comm.all_reduce(inp)
            torch.ops._C_sm70_hier_ar.sm70_hier_ar_timer(ts_out)
            return out

        for _ in range(3):  # warm-up outside capture (NCCL lazy init,
            body()          # first protocol epochs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=pool):
            out = body()
        return {"graph": g, "out": out,
                "ts_in": ts_in, "ts_out": ts_out}

    def replay_once(cap: dict) -> float:
        cap["graph"].replay()
        torch.cuda.synchronize()  # wait for the graph (incl. timer writes)
        # us over the in-graph AR sub-span (device-side ns timer)
        return (int(cap["ts_out"].item())
                - int(cap["ts_in"].item())) / 1000.0

    g2: dict = {}
    rounds = args["rounds"]
    for m in args["shapes"]:
        cap_a = capture_graph(comm_a, m)
        cap_b = capture_graph(comm_b, m)
        ts_a: list[float] = []
        ts_b: list[float] = []
        for _ in range(rounds):  # interleaved (A,B,A,B,...), barrier-gated
            ts_a.append(replay_once(cap_a))
            dist.barrier(group)
            ts_b.append(replay_once(cap_b))
            dist.barrier(group)
        del cap_a["graph"], cap_b["graph"]
        torch.cuda.empty_cache()
        all_a: list = [None] * gw
        all_b: list = [None] * gw
        dist.all_gather_object(all_a, ts_a, group=group)
        dist.all_gather_object(all_b, ts_b, group=group)
        if rank == 0:
            a = np.concatenate(all_a)
            b = np.concatenate(all_b)
            g2[m] = {
                "per_call_median_us": {"a": float(np.median(a)),
                                       "b": float(np.median(b))},
                "per_call_p95_us": {"a": float(np.percentile(a, 95)),
                                    "b": float(np.percentile(b, 95))},
                "n_samples": int(a.size),
            }
        dist.barrier(group)
    results["g2"] = g2 if rank == 0 else None

    # ------------------------------------------------------------------
    # Skew table (steady state, >=1e4 collectives/arm): graph of SK_AR
    # ARs x SK_REP replays. Per rank per replay: t_in = CPU monotonic at
    # replay launch (input-ready), t_lat = device-side globaltimer span
    # over the whole AR chain / sk_ar (in-graph events unreadable from
    # the host on this stack — see G2).
    # Cross-rank: input-ready skew + end-after-all-inputs-ready.
    # ------------------------------------------------------------------
    skew: dict = {}
    if args["skew"]:
        sk_ar, sk_rep = 40, 250  # 10,000 collectives/arm (1Cat: 10,788)
        for arm, comm in (("a", comm_a), ("b", comm_b)):
            dist.barrier(group)  # same arm on every rank (ordering rule)
            inp = make_inputs(1, seed=7)
            t_in_dev = torch.zeros(1, dtype=torch.int64, device=device)
            t_out_dev = torch.zeros(1, dtype=torch.int64, device=device)

            def body_sk():
                _ = dummy.mul(1.0)
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_timer(t_in_dev)
                x = inp
                for _ in range(sk_ar):
                    x = comm.all_reduce(x)
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_timer(t_out_dev)
                return x

            for _ in range(2):
                body_sk()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(g, pool=pool):
                body_sk()
            t_in: list[float] = []
            t_lat: list[float] = []
            for _ in range(sk_rep):
                t0 = time.monotonic()
                g.replay()
                torch.cuda.synchronize()
                t_in.append(t0)
                t_lat.append((int(t_out_dev.item())
                              - int(t_in_dev.item())) / 1000.0 / sk_ar)
            del g
            ins, lats = [None] * gw, [None] * gw
            dist.all_gather_object(ins, t_in, group=group)
            dist.all_gather_object(lats, t_lat, group=group)
            if rank == 0:
                ins = np.array(ins).T   # [rep][rank]
                lats = np.array(lats).T
                t_in_skew_us = float(np.max(
                    ins.max(axis=1) - ins.min(axis=1)))
                end_all_us = float(np.max(
                    (ins * 1e6 + lats).max(axis=1) - (ins * 1e6).max(axis=1)))
                skew[arm] = {
                    "n_replays": int(sk_rep),
                    "collectives": int(sk_rep * sk_ar),
                    "input_ready_skew_us_max": t_in_skew_us,
                    "ar_span_median_us": float(np.median(lats)),
                    "ar_span_max_us": float(np.max(lats)),
                    # collective end after ALL inputs ready (cross-rank):
                    "end_after_all_inputs_us": end_all_us,
                }
            dist.barrier(group)
    results["skew"] = skew if rank == 0 else None

    # ------------------------------------------------------------------
    # G4 stress + determinism (B only): N-collective pure-graph replay;
    # then a second identical replay -> byte-identical output.
    # ------------------------------------------------------------------
    g4: dict = {}
    if args["stress"]:
        dist.barrier(group)
        inp = make_inputs(1, seed=4242)

        def body_st():
            x = inp
            for _ in range(args["stress"]):
                x = comm_b.all_reduce(x)
            return x

        for _ in range(2):
            body_st()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=pool):
            out_st = body_st()
        dist.barrier(group)
        t0 = time.monotonic()
        g.replay()
        torch.cuda.synchronize()
        wall_s = time.monotonic() - t0
        b1 = out_st.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        g.replay()
        torch.cuda.synchronize()
        b2 = out_st.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        del g
        if rank == 0:
            g4 = {
                "n_collectives": args["stress"],
                "wall_s": wall_s,
                "us_per_call": wall_s * 1e6 / args["stress"],
                "rerun_byte_identical": bool(b1 == b2),
            }
            print(f"[worker] rank0: G4 stress {args['stress']} ARs in "
                  f"{wall_s:.2f}s ({g4['us_per_call']:.2f} us/call), "
                  f"rerun byte-identical={g4['rerun_byte_identical']}",
                  flush=True)
        dist.barrier(group)
    results["g4"] = g4 if rank == 0 else None

    # ------------------------------------------------------------------
    # G6 VRAM: per-arm torch-allocator peak; the route's raw cudaMalloc
    # IPC buffer (outside the torch allocator) is gathered too, with its
    # sizing class — the verdict gates it (2.2 MiB mode-1 bound; the
    # 1.5 GB working-set cap counts the raw bytes for the mode-2/file-set
    # bench sizing).
    # ------------------------------------------------------------------
    def peak_arm(comm, m: int) -> int:
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(50):
            comm.all_reduce(make_inputs(m, seed=55))
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated(device))

    peak_a = peak_arm(comm_a, 1)
    dist.barrier(group)
    peak_b = peak_arm(comm_b, 1)
    dist.barrier(group)
    g6: dict = {}
    # all_gather_object is a COLLECTIVE: every rank must call it (F3b rule —
    # a collective guarded by a per-rank branch deadlocks: rank 0 waits in
    # it forever while ranks 1..7 race to the next barrier; caught live in
    # smoke 6d). Rank 0 alone derives the verdict fields from the peaks.
    peaks = [None] * gw
    dist.all_gather_object(
        peaks, [peak_a, peak_b,
                int(getattr(route_b, "_alloc_bytes", 0)),
                bool(getattr(route_b, "_sized_prefill", False))],
        group=group)
    if rank == 0:
        pa = np.array([p[0] for p in peaks])
        pb = np.array([p[1] for p in peaks])
        rb = int(peaks[0][2])
        g6 = {
            "torch_peak_a_bytes": [int(x) for x in pa],
            "torch_peak_b_bytes": [int(x) for x in pb],
            "delta_b_minus_a_max_bytes": int((pb - pa).max()),
            "route_ipc_bytes_per_rank": rb,
            "route_ipc_sized_prefill": bool(peaks[0][3]),
            "working_set_ok_per_rank": [
                bool(max(pa[i], pb[i]) + rb <= WORKING_SET_CAP_BYTES)
                for i in range(WORLD)],
        }
    results["g6"] = g6 if rank == 0 else None

    if rank == 0:
        out = Path(args["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "rank0.json").write_text(json.dumps(results, indent=1))
        print("[worker] rank0: results written", flush=True)
    dist.barrier(group)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-m", type=int, default=None,
                    help="verify-batch shape M (default: derived, O1)")
    ap.add_argument("--rounds", type=int, default=300,
                    help="per-shape per-arm interleaved replay rounds")
    ap.add_argument("--stress", type=int, default=8700,
                    help="G4 N-collective stress (1Cat: 8,700)")
    ap.add_argument("--no-skew", action="store_true",
                    help="skip the >=1e4-collective skew table")
    ap.add_argument("--no-g0", action="store_true",
                    help="SKIP window guards (dry runs only — verdict "
                         "then INVALID)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE (non-gating): the FIRST real GPU launch "
                         "on this box must be --smoke (R3 lesson) — M=1 "
                         "only, rounds<=10, no stress, no skew; the "
                         "verdict is labeled 'SMOKE (non-gating)' and is "
                         "never a gate-1 verdict")
    return ap


def _run_config(smoke: bool, rounds: int, stress: int | None,
                no_skew: bool) -> dict:
    """Run config (pure — CPU-checkable by the selftest). `shapes` is
    [1] in smoke, else "derive" (main fills it in from the O1 verify-M)."""
    if smoke:
        return {"shapes": [1], "rounds": min(rounds, 10),
                "stress": 0, "skew": False}
    return {"shapes": "derive", "rounds": rounds,
            "stress": 0 if stress is None else stress,
            "skew": not no_skew}


def main() -> int:
    import os
    args = build_parser().parse_args()

    if args.no_g0:
        window = {"window_day": time.strftime("%Y-%m-%d", time.localtime()),
                  "no_g0": True, "xids_before": _dmesg_xids()}
        print("[G0] SKIPPED (--no-g0): this run's verdict is INVALID",
              file=sys.stderr)
    else:
        window = check_window_g0()

    cfg = _run_config(args.smoke, args.rounds, args.stress, args.no_skew)
    if args.smoke:
        verify_m, how = 1, "smoke: M=1 only (non-gating)"
        shapes = cfg["shapes"]
    else:
        verify_m, how = (args.verify_m, "user override") if args.verify_m \
            else derive_verify_m()
        shapes = sorted({1, verify_m, 8192})
    rounds, stress = cfg["rounds"], cfg["stress"]
    skew_on = cfg["skew"]
    out = Path(args.out) if args.out else (
        RESULTS_DIR / f"ar_hier_gate1_{window['window_day'].replace('-', '')}")

    print(f"[gate1] shapes M={shapes} (verify={verify_m}: {how}); "
          f"rounds={rounds}, stress={stress}, "
          f"skew={'off' if not skew_on else 'on'}"
          f"{'; SMOKE (non-gating)' if args.smoke else ''}", flush=True)

    import torch.multiprocessing as mp
    port = 29617 + (int(time.time()) % 100)
    ctx = mp.get_context("spawn")
    workers = []
    for rank in range(WORLD):
        w = ctx.Process(target=worker,
                        args=(rank, port, {
                            "shapes": shapes, "rounds": rounds,
                            "stress": stress,
                            "skew": skew_on,
                            "out": out,
                        }), daemon=False)
        w.start()
        workers.append(w)

    # Hang discipline (HW_GUIDE §3.5 / spec R6): a stuck device-side wait
    # never raises — the parent enforces a ceiling, waits for self-abort,
    # then TERM one by one (NO kill -9). Ceiling covers the full program:
    # skew (~2 arms x 10^4 ARs) dominates, ~10 min at 30 us/AR.
    HANG_CEILING_S = 1500.0
    t0 = time.monotonic()
    hung = False
    while any(w.is_alive() for w in workers):
        for w in workers:
            w.join(timeout=5.0)
        if time.monotonic() - t0 > HANG_CEILING_S:
            print("[gate1] HANG past the ceiling — waiting 90 s for "
                  "self-abort, then TERM one by one (NO kill -9)",
                  flush=True)
            time.sleep(90.0)
            hung = True
            break
    for w in workers:
        if w.is_alive():
            w.terminate()
            w.join(timeout=60)
    exit_codes = [w.exitcode for w in workers]

    if hung or any(c not in (0, None) for c in exit_codes):
        out.mkdir(parents=True, exist_ok=True)
        (out / "rank0.json").write_text(json.dumps(
            {"error": "hang or worker failure",
             "exit_codes": exit_codes}))
        _verdict(out, window, shapes, forced_error=True, smoke=args.smoke)
        return 1

    return _verdict(out, window, shapes, smoke=args.smoke)["exit_code"]


def _verdict(out: Path, window: dict, shapes: list[int],
             forced_error: bool = False, smoke: bool = False) -> dict:
    """rank0.json -> pre-registered gate verdict (spec §4). A smoke run
    is SMOKE (non-gating): G4 is SKIPPED (the 8,700-collective stress is
    exactly what --smoke must not run) and the verdict line is labeled
    'SMOKE (non-gating)' — never a plain PASS/FAIL gate verdict."""
    import numpy as np
    try:
        rec = json.loads((out / "rank0.json").read_text())
    except OSError:
        rec = {}
    v: dict = {"window": window, "gates": {}, "exit_code": 1 if forced_error
               else 2}
    if forced_error or "error" in rec:
        err = rec.get("error", "worker failure")
        v["gates"] = {"G0": f"FAIL: {err}"}
        v["verdict"] = (f"{'SMOKE (non-gating) FAIL' if smoke else 'FAIL'}"
                        f" — {err} (window invalid)")
        _report(v, out)
        return v

    # JSON round-trip coerces the int shape keys to str (json.loads never
    # restores int keys) — normalize back to int so the lookups below
    # (g1.get(m) / med() with m = int) hit the per-shape rows.
    g1 = {int(k): v for k, v in (rec.get("g1") or {}).items()}
    g2 = {int(k): v for k, v in (rec.get("g2") or {}).items()}
    g4, g6 = rec.get("g4") or {}, rec.get("g6") or {}
    skew = rec.get("skew") or {}
    verify_m = (max(s for s in shapes if s != 8192) if 8192 in shapes
                else max(shapes))

    # G1 (HARD)
    err_ok = all(
        g1.get(m, {}).get("max_abs_b", float("inf"))
        <= g1.get(m, {}).get("max_abs_a", 0.0)
        for m in shapes)
    v["gates"]["G1"] = "PASS" if (
        all(g1.get(m, {}).get("b_bitwise_equal_across_ranks", False)
            for m in shapes)
        and all(g1.get(m, {}).get("b_finite", False) for m in shapes)
        and err_ok) else "FAIL"
    v["g1_detail"] = g1

    # G2 (HARD): primary OR alternate
    def med(tag: str, m: int) -> float:
        return (g2.get(m) or {}).get("per_call_median_us", {}) \
            .get(tag, float("inf"))

    # Pure-wire lower bound = slowest route leg at the payload size
    # (spec §1.3 table: NV1 23.0 GB/s mirror leg dominates at these
    # sizes; bytes / (GB/s) = us):
    def wire_lb_us(m: int) -> float:
        return (m * HIDDEN * 2) / 23.0e3

    primary = (med("b", 1) <= 0.9 * med("a", 1)
               and med("b", verify_m) <= 0.9 * med("a", verify_m))
    # Alternate band (reviewer flag, report-only — no behavior change):
    # the 1.0 us coordination proxy and the spec §1.3 leg-table wire lower
    # bound are PRE-REGISTERED before any run (spec §4 G2), not fitted to
    # this run's measurements.
    coord = {m: med("b", m) - wire_lb_us(m) for m in (1, verify_m)}
    end_b = (skew.get("b") or {}).get("end_after_all_inputs_us",
                                      float("inf"))
    end_a = (skew.get("a") or {}).get("end_after_all_inputs_us",
                                      float("inf"))
    alternate = bool(coord) and all(c <= 1.0 for c in coord.values()) \
        and end_b < end_a
    v["gates"]["G2"] = "PASS" if (primary or alternate) else "FAIL"
    v["g2_detail"] = {**g2, "skew": skew,
                      "wire_lb_us": {m: wire_lb_us(m)
                                     for m in (1, verify_m)},
                      "b_coordination_us": coord,
                      "end_after_all_inputs_us": {"a": end_a, "b": end_b},
                      "primary": primary, "alternate": alternate}

    # G3 (INFO)
    m8 = 8192
    if m8 in shapes:
        v["gates"]["G3"] = "INFO"
        v["g3_detail"] = {
            "ratio_b_over_a": (med("b", m8) / med("a", m8)
                               if med("a", m8) else None),
            "note": "expected >= 1.4 (hierarchical loses -> routing rule "
                    "confirmed); a surprise win is REPORTED, not "
                    "auto-adopted",
        }
    else:
        v["gates"]["G3"] = "SKIPPED (no M=8192 arm)"
        v["g3_detail"] = {"note": "M=8192 arm not run in this config"}

    # G4 (HARD; SKIPPED in smoke — the 8,700-collective stress is exactly
    # what --smoke must not run; a skipped gate is not counted in all_pass)
    if smoke and not g4:
        v["gates"]["G4"] = "SKIPPED (smoke)"
        v["g4_detail"] = {"note": "--smoke: stress=0, G4 not evaluated"}
    else:
        v["gates"]["G4"] = "PASS" if (
            g4.get("rerun_byte_identical", False)
            and 0 < g4.get("wall_s", float("inf")) < 600.0) else "FAIL"
        v["g4_detail"] = g4

    # G5 (HARD)
    new_xids = [x for x in _dmesg_xids()
               if x not in window.get("xids_before", [])]
    v["gates"]["G5"] = "PASS" if not new_xids else "FAIL"
    v["g5_detail"] = {"new_xid_lines": new_xids}

    # G6 (HARD)
    d = g6.get("delta_b_minus_a_max_bytes", float("inf"))
    ws_ok = g6.get("working_set_ok_per_rank", [False] * WORLD)
    # The route's raw-IPC buffer (outside the torch allocator): the
    # production mode-1 class is bounded by the pre-registered constant
    # (2.2 MiB); the mode-2/file-set bench sizing (~335.5 MiB, arm B) is
    # allowed only inside the 1.5 GB working-set cap guard above (which
    # counts the raw route bytes per rank).
    rpc = g6.get("route_ipc_bytes_per_rank", float("inf"))
    ipc_ok = all(ws_ok) if g6.get("route_ipc_sized_prefill") \
        else rpc <= G6_ROUTE_IPC_MODE1_MAX_BYTES
    v["gates"]["G6"] = "PASS" if (d <= 0 and all(ws_ok) and ipc_ok) \
        else "FAIL"
    v["g6_detail"] = {
        **g6,
        "g6_route_ipc_mode1_bound_bytes": G6_ROUTE_IPC_MODE1_MAX_BYTES,
        "g6_route_ipc_within_bound": bool(ipc_ok),
    }

    hard = ("G1", "G2", "G4", "G5", "G6")
    evaluated = {g: v["gates"].get(g) for g in hard
                 if v["gates"].get(g) != "SKIPPED (smoke)"}
    all_pass = all(s == "PASS" for s in evaluated.values())
    if smoke:
        v["verdict"] = (
            "SMOKE (non-gating) PASS — first-launch sanity OK (M=1, "
            "rounds<=10, no stress/skew; G4 SKIPPED); NOT a gate-1 "
            "verdict — re-run without --smoke for the real window"
            if all_pass else
            "SMOKE (non-gating) FAIL — a HARD gate failed in the "
            "first-launch sanity run (G4 SKIPPED); fix, then re-run "
            "without --smoke for the real window")
    else:
        v["verdict"] = (
            "PASS — shadow direction wired-verified -> round-3 online A/B "
            "authorized (one restart, 11673, same-window arm A current "
            "boot / arm B shadow-on boot, ruler "
            "sm70_tests/baseline.py 9 buckets, decode primary)"
            if all_pass else
            "FAIL — any HARD fail = rollback; the mechanism is recorded "
            "rejected with the per-shape breakdown (spec §4 verdict "
            "vocabulary); G2-fail-only on the transport term -> the "
            "graph-scheduling component (d) is re-tested standalone")
    v["exit_code"] = 0 if all_pass else 2
    if smoke:
        v["smoke"] = True
    if window.get("no_g0"):
        v["verdict"] += " (INVALID: G0 was skipped)"
    _report(v, out)
    return v


def _report(v: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate1.json").write_text(json.dumps(v, indent=1))
    print("\n================ GATE-1 VERDICT ================")
    for k in ("G0", "G1", "G2", "G3", "G4", "G5", "G6"):
        if k in v["gates"]:
            print(f"  {k}: {v['gates'][k]}")
    g2 = v.get("g2_detail") or {}
    for m, row in (g2.items() if isinstance(g2, dict) else []):
        if isinstance(row, dict) and "per_call_median_us" in row:
            print(f"  {m}: A={row['per_call_median_us']['a']:.3f} us, "
                  f"B={row['per_call_median_us']['b']:.3f} us")
    print(f"  VERDICT: {v['verdict']}")
    print(f"  results: {out / 'gate1.json'}")


if __name__ == "__main__":
    sys.exit(main())
