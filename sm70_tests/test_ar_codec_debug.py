#!/usr/bin/env python3
"""L1 wire-codec DEBUG UNIT (systematic-debugging Phase 1/2 evidence).

Context: window #3 (2026-10-15) broke content on the 4-bit wires on the
hierarchical route — e4m3 (wire=1): battery 3/6 + decode 40 tps (vs fp32
65 / 15.5ms: +9.4ms TPOT); nvfp4 (wire=2): battery 0/6, decode 63 tps
(flat). Per user ruling the 4-bit AR is content-safe in other projects, so
this unit first separates IMPLEMENTATION BUG from intrinsic codec precision,
and measures the kernel per-call cost per wire (attributes the +9.4ms to
kernel vs model-level effect).

Method (8 ranks on the 8 V100s, quiet window, serve DOWN):
  * the REAL op through the REAL service seam (CudaCommunicator.all_reduce,
    mode 2 / prefill-sized buffers, the exact dispatch path),
  * fixed per-rank seeded fp16 inputs (m in {1, 8, 64} -> n in
    {2560, 20480, 163840} elements),
  * rank-0 builds an FP32-FAITHFUL model of the whole pipeline
    (quad partials in kernel accumulation order -> per-codec quantize in
    fp32 ops -> dequant fp32 -> fp32 add of the two quad partials ->
    __float2half_rn) and compares BIT-EXACTLY with the kernel output:
      - codec functions: shared with sm70_tests/test_ar_wire_codecs.py
        (bit-exact references; the fp32-faithful wrappers here mirror the
        device arithmetic op-for-op: __fdiv_rn = np.float32 div, etc.)
      - kernel == model bitwise across all 8 ranks
        -> codec implemented correctly; content breakage = intrinsic
        precision on this stream (L1 precision ruling)
      - mismatch -> n_mismatch pattern localizes the implementation bug
        (scale-index/slot-stride/offset class)
  * per-wire kernel cost: CUDA-graph capture of a single AR bracketed by
    the in-graph sm70_hier_ar_timer() device globaltimer (same technique
    as bench_ar_hier_gate1 G2), 50 replays, per-rank median, at m=1 (decode
    step: n=2560) and m=8 (verify scale).

Run (parent entry; spawns 8 workers; quiet window enforced):
    source /data/src/sm70_env.sh
    /data/vllm-dev/bin/python /data/src/vllm-v2/sm70_tests/test_ar_codec_debug.py \
        [--out /data/sm70_results/round5/ar_codec_debug]
Exit: 0 = all wire modes model-bit-exact + perf table collected;
      2 = any mismatch (debug evidence written); 1 = window/refusal/hang.
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
WORLD = 8
HIDDEN = 2560
# m -> elements (m x HIDDEN fp16); all multiples of 128 (wire groups)
SHAPES = [1, 8, 64]
WIRES = [0, 1, 2, 3]
PERF_REPLAYS = 50
PERF_SHAPES = [1, 8]


# ---------------------------------------------------------------------------
# G0 window discipline (same pattern as bench_ar_hier_gate1)
# ---------------------------------------------------------------------------


def _journal_last_age_s() -> float | None:
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
        if ts.tzinfo is None:
            return time.time() - time.mktime(ts.timetuple())
        return time.time() - ts.timestamp()
    except Exception:  # noqa: BLE001 - fail closed
        return None


def check_window_g0() -> dict:
    age = _journal_last_age_s()
    if age is None:
        raise SystemExit("G0 FAIL: serve journal unverifiable — refuse")
    if age < 60.0:
        raise SystemExit(
            f"G0 FAIL: journal last line {age:.0f}s ago — need >= 60s silence")
    return {"journal_silence_s": age,
            "window_day": time.strftime("%Y-%m-%d", time.localtime())}


# ---------------------------------------------------------------------------
# FP32-faithful codec model (op-for-op mirror of sm70_hier_ar.cu)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_ar_wire_codecs import (  # noqa: E402
    NV4_GRID, _e2m1_code, e4m3_to_f32, f32_to_e4m3,
)

NV4_BLK = 16
E4M3_BLK = 128
NV4_GRID32 = np.array(NV4_GRID, dtype=np.float32)


def _f32(x) -> np.float32:
    return np.float32(x)


def _h2f(x: np.ndarray) -> np.ndarray:
    """fp16 -> fp32 (exact)."""
    return x.astype(np.float32)


def _quad_partial(xs: list[np.ndarray]) -> np.ndarray:
    """Kernel step (4): FP32 accumulation in global-rank order over the 4
    quad inputs (quad_in_buf order = ascending rank)."""
    acc = _h2f(np.asarray(xs[0], dtype=np.float16)).copy()
    for k in range(1, 4):
        acc = acc + _h2f(np.asarray(xs[k], dtype=np.float16))
    return acc


def _codec_e4m3(p: np.ndarray) -> np.ndarray:
    """Wire=1: per-128 e4m3 (writer: mx/448 fp32 div, code of v/sc fp32 div;
    reader: e4m3_to_f32 * sc fp32 mul)."""
    n = p.shape[0]
    out = np.empty(n, dtype=np.float32)
    for g in range(n // E4M3_BLK):
        blk = p[g * E4M3_BLK:(g + 1) * E4M3_BLK]
        mx = np.abs(blk).max()
        sc = _f32(mx / _f32(448.0))
        if sc == 0.0:
            sc = _f32(1.0)
        for j in range(E4M3_BLK):
            v = _f32(blk[j] / sc)  # fp32 div (kernel: v / sc); blk is the slice
            code = f32_to_e4m3(float(v))
            out[g * E4M3_BLK + j] = _f32(e4m3_to_f32(code) * sc)
    return out


def _codec_nv4(p: np.ndarray) -> np.ndarray:
    """Wire=2: per-16 e2m1 + e4m3 scale (s=e4m3(mx/6) RN fp32 div,
    inv=__fdiv_rn(1,s), code on |x|*inv fp32 mul; reader: grid*s fp32 mul)."""
    n = p.shape[0]
    out = np.empty(n, dtype=np.float32)
    for g in range(n // NV4_BLK):
        blk = p[g * NV4_BLK:(g + 1) * NV4_BLK]
        mx = np.abs(blk).max()
        sbyte = f32_to_e4m3(float(_f32(mx / _f32(6.0))) if mx > 0.0
                            else 1.0)
        s = _f32(e4m3_to_f32(sbyte))
        inv = _f32(_f32(1.0) / s) if s > 0.0 else _f32(1.0)
        for j in range(NV4_BLK):
            x = blk[j]  # blk is the slice (blk[g*NV4_BLK+j] = 03:2x run IndexError)
            code = _e2m1_code(float(_f32(np.abs(x) * inv)))
            v = _f32(NV4_GRID32[code] * s)
            out[g * NV4_BLK + j] = -v if (x < 0.0 and code) else v
    return out


def _codec_fp16(p: np.ndarray) -> np.ndarray:
    """Wire=3: __float2half_rn (IEEE RNE; np.float16 cast verified RNE in
    test_ar_wire_codecs), then exact half->fp32."""
    return p.astype(np.float16).astype(np.float32)


CODECS = {0: lambda p: p, 1: _codec_e4m3, 2: _codec_nv4, 3: _codec_fp16}


def model_output(wire: int, inputs8: list[np.ndarray]) -> np.ndarray:
    """Full-pipeline model: quad partials -> codec (both quads, rank-
    replicated) -> fp32 add of the two operands -> __float2half_rn."""
    p0 = _quad_partial(inputs8[:4])
    p1 = _quad_partial(inputs8[4:])
    deq = CODECS[wire]
    tot = deq(p0) + deq(p1)
    return tot.astype(np.float16)


# ---------------------------------------------------------------------------
# Worker (one per rank)
# ---------------------------------------------------------------------------


def worker(rank: int, port: int, out: str) -> None:
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    os.environ["VLLM_SM70_AR_HIER"] = "2"  # mode 2: prefill-sized buffers
    os.environ["VLLM_SM70_AR_HIER_MAX_BYTES"] = str(64 * HIDDEN * 2)
    os.environ.pop("VLLM_SM70_AR_HIER_FILE", None)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # Finite collective timeout: a dead peer must fail the survivors fast
    # (this build's gloo wait busy-spins; 24h = the run we had to kill).
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=WORLD,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=600))
    group = dist.new_group(backend="gloo")
    gw = dist.get_world_size(group)

    import vllm  # noqa: F401 (loads _C_stable_libtorch)
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    comm = CudaCommunicator(
        group, device=device, unique_name="tp:0",
        global_ranks=list(range(WORLD)), global_world_size=WORLD)
    route = comm.sm70_hier_ar
    active = route is not None and not getattr(route, "disabled", True)
    dist.barrier(group)
    if rank == 0:
        print(f"[unit] route {'ACTIVE' if active else 'FAILED-CLOSED'} "
              f"n_max={getattr(route, '_n_max', 0)}", flush=True)
    if not active:
        _write(out, {"error": "route failed closed"}, rank=rank)
        dist.destroy_process_group()
        return

    def make_inputs(m: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        t = torch.randn(m, HIDDEN, generator=g, dtype=torch.float32)
        return (t / 8.0).to(device, dtype=torch.float16)

    rec: dict = {"rank": rank}
    mismatches: list = []

    # ---- bit-exact model check (all ranks call in identical order) ----
    for m in SHAPES:
        inp = make_inputs(m, seed=42 + rank)
        for w in WIRES:
            route.wire = w  # lockstep: same value on every rank
            out_t = route.all_reduce(inp)
            dist.barrier(group)
            inp_b = inp.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            out_b = out_t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            inps, outs = [None] * gw, [None] * gw
            dist.all_gather_object(inps, inp_b, group=group)
            dist.all_gather_object(outs, out_b, group=group)
            dist.barrier(group)
            if rank == 0:
                in8 = [np.frombuffer(b, dtype=np.float16).reshape(-1)
                       for b in inps]
                out8 = [np.frombuffer(b, dtype=np.float16) for b in outs]
                ref = model_output(w, in8)
                p0_dbg = _quad_partial([np.asarray(x, np.float16)
                                        for x in in8[:4]])
                # outs are bytes (gathered tobytes) — out8 is the decoded form
                bitwise_ranks = all(o.tobytes() == out8[0].tobytes()
                                    for o in out8)
                mm = [int((o != ref).sum()) for o in out8]
                # Per-element pattern dumps (mismatch attribution, #11931):
                # kernel vs model, both fp16 bits; inputs of all 8 ranks.
                try:
                    np.save(f"{out}/dump_w{w}_m{m}_in.npy",
                            np.concatenate(in8))
                    np.save(f"{out}/dump_w{w}_m{m}_ker.npy",
                            np.asarray(out8[0], dtype=np.float16))
                    np.save(f"{out}/dump_w{w}_m{m}_mdl.npy",
                            np.asarray(ref, dtype=np.float16))
                except Exception:
                    pass
                # ---- region attribution dump (wire=1 only, rank 0):
                # read the slot the op just used; compare staging / payload
                # codes / scales against the model to localize the residual
                # mismatch (step-4 staging vs writer encoder vs reader).
                if w == 1:
                    try:
                        k = (SHAPES.index(m) * len(WIRES) + w)
                        slot = (k + 1) & 1  # E = calls-so-far, s = E & 1
                        nm = route._n_max
                        stg = torch.empty(nm, dtype=torch.float32,
                                          device=device)
                        pay = torch.empty(nm, dtype=torch.uint8,
                                          device=device)
                        scf = torch.empty((nm + 127) // 128,
                                          dtype=torch.float32, device=device)
                        scb = torch.empty((nm + 15) // 16,
                                          dtype=torch.uint8, device=device)
                        torch.ops._C_sm70_hier_ar.sm70_hier_ar_dump(
                            route._fa, slot, stg, pay, scf, scb)
                        torch.cuda.synchronize()
                        n = m * HIDDEN
                        stg_n = stg[:n].cpu().numpy()
                        pay_n = pay[:n].cpu().numpy()
                        scf_n = scf[:(n // 128)].cpu().numpy()
                        st_mm = int(np.sum(
                            stg_n.view(np.float32)
                            != p0_dbg.astype(np.float32)))
                        # model codes/scales (own quad = p0 for rank 0)
                        from test_ar_wire_codecs import f32_to_e4m3
                        msc = np.empty(n // 128, dtype=np.float32)
                        mcode = np.empty(n, dtype=np.uint8)
                        for gg in range(n // 128):
                            blk = p0_dbg[gg * 128:(gg + 1) * 128]
                            sc = np.float32(
                                np.float32(np.abs(blk).max())
                                / np.float32(448.0))
                            if sc == 0.0:
                                sc = np.float32(1.0)
                            msc[gg] = sc
                            for j in range(128):
                                mcode[gg * 128 + j] = f32_to_e4m3(
                                    float(np.float32(blk[j] / sc)))
                        code_mm = int(np.sum(pay_n != mcode))
                        sc_mm = int(np.sum(scf_n != msc))
                        print(f"[unit-dump] m={m}: OWN staging_mm={st_mm}/{n} "
                              f"code_mm={code_mm}/{n} scale_mm={sc_mm}/{n // 128}",
                              flush=True)
                        np.save(f"{out}/dump_w1_m{m}_stg.npy", stg_n)
                        np.save(f"{out}/dump_w1_m{m}_pay.npy", pay_n)
                        np.save(f"{out}/dump_w1_m{m}_scf.npy", scf_n)
                        # ---- mirror side (quad1): staging/payload/scales ----
                        stg2 = torch.empty(nm, dtype=torch.float32,
                                           device=device)
                        pay2 = torch.empty(nm, dtype=torch.uint8,
                                           device=device)
                        scf2 = torch.empty((nm + 127) // 128,
                                           dtype=torch.float32, device=device)
                        scb2 = torch.empty((nm + 15) // 16,
                                           dtype=torch.uint8, device=device)
                        torch.ops._C_sm70_hier_ar.sm70_hier_ar_dump_mr(
                            route._fa, slot, stg2, pay2, scf2, scb2)
                        torch.cuda.synchronize()
                        stg2n = stg2[:n].cpu().numpy()
                        pay2n = pay2[:n].cpu().numpy()
                        scf2n = scf2[:(n // 128)].cpu().numpy()
                        p1_dbg = _quad_partial([np.asarray(x, np.float16)
                                                for x in in8[4:8]])
                        st_mm2 = int(np.sum(
                            stg2n != p1_dbg.astype(np.float32)))
                        msc1 = np.empty(n // 128, dtype=np.float32)
                        mcode1 = np.empty(n, dtype=np.uint8)
                        for gg in range(n // 128):
                            blk = p1_dbg[gg * 128:(gg + 1) * 128]
                            sc = np.float32(
                                np.float32(np.abs(blk).max())
                                / np.float32(448.0))
                            if sc == 0.0:
                                sc = np.float32(1.0)
                            msc1[gg] = sc
                            for j in range(128):
                                mcode1[gg * 128 + j] = f32_to_e4m3(
                                    float(np.float32(blk[j] / sc)))
                        code_mm2 = int(np.sum(pay2n != mcode1))
                        sc_mm2 = int(np.sum(scf2n != msc1))
                        print(f"[unit-dump] m={m}: MR staging_mm={st_mm2}/{n} "
                              f"code_mm={code_mm2}/{n} scale_mm={sc_mm2}/{n // 128}",
                              flush=True)
                        np.save(f"{out}/dump_w1_m{m}_stg2.npy", stg2n)
                        np.save(f"{out}/dump_w1_m{m}_pay2.npy", pay2n)
                        np.save(f"{out}/dump_w1_m{m}_scf2.npy", scf2n)
                        # ---- CPU reconstruction from dumped payloads:
                        # deq(own)*sc + deq(mirror)*sc -> h2f; decide which
                        # side the kernel's residual lives in.
                        def _deq_block(pay_x, scf_x):
                            o = np.empty(n, dtype=np.float32)
                            for gg in range(n // 128):
                                sc = np.float32(scf_x[gg])
                                for j in range(128):
                                    o[gg * 128 + j] = np.float32(
                                        np.float32(
                                            e4m3_to_f32(
                                                int(pay_x[gg * 128 + j])))
                                        * sc)
                            return o
                        rec2 = (_deq_block(pay_n, scf_n)
                                + _deq_block(pay2n, scf2n)
                                ).astype(np.float16)
                        ker_n = np.asarray(out8[0], dtype=np.float16)
                        mdl_n = np.asarray(ref, dtype=np.float16)
                        r_ker = int(np.sum(rec2 != ker_n))
                        r_mdl = int(np.sum(rec2 != mdl_n))
                        print(f"[unit-dump] m={m}: recon vs ker={r_ker}/{n} "
                              f"vs mdl={r_mdl}/{n}", flush=True)
                        # (post-barrier re-dump removed: the dump already runs
                        # after the all-ranks barrier at the case tail, so a
                        # re-dump reads the same final state — and a barrier
                        # HERE would deadlock: this block is rank0-only while
                        # the other 7 ranks race ahead to _self_attr's
                        # all_gather (rerun9 hang, 2026-10-15).)
                    except Exception as e:  # attribution must never kill the unit
                        import traceback
                        print(f"[unit-dump] m={m}: FAILED {e!r}\n"
                              + traceback.format_exc(), flush=True)
                # (all-rank _self_attr removed 2026-10-15: it sat INSIDE
                #  the rank==0 gate -> rank0-only all_gather/barrier =
                #  guaranteed 7-rank deadlock (rerun9/12 hangs). Its
                #  question (who wrote the stale half) is answered by
                #  the in_stride layout bug, #11938.)
                row = {
                    "wire": w, "m": m, "n": m * HIDDEN,
                    "bitwise_across_ranks": bool(bitwise_ranks),
                    "model_bitwise": bool(max(mm) == 0),
                    "n_mismatch": max(mm),
                    "max_abs_err": float(np.max(
                        np.abs(out8[0].astype(np.float32)
                                 - ref.astype(np.float32)))),
                }
                if not (row["bitwise_across_ranks"]
                        and row["model_bitwise"]):
                    mismatches.append(row)
                print(f"[unit] m={m:3d} wire={w}: "
                      f"ranks-eq={row['bitwise_across_ranks']} "
                      f"model-eq={row['model_bitwise']} "
                      f"n_mismatch={row['n_mismatch']} "
                      f"max_abs_err={row['max_abs_err']:.3e}", flush=True)
            rec.setdefault("model", {})[f"{m}/{w}"] = (
                row if rank == 0 else None)
    route.wire = 0

    # ---- per-wire kernel cost (graph-captured, in-graph device timer) ----
    perf: dict = {}
    for w in WIRES:
        for m in PERF_SHAPES:
            dist.barrier(group)
            route.wire = w
            inp = make_inputs(m, seed=99)
            ts_in = torch.zeros(1, dtype=torch.int64, device=device)
            ts_out = torch.zeros(1, dtype=torch.int64, device=device)

            def body():
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_timer(ts_in)
                x = route.all_reduce(inp)
                torch.ops._C_sm70_hier_ar.sm70_hier_ar_timer(ts_out)
                return x

            for _ in range(3):
                body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(g, pool=pool):
                body()
            ts: list[float] = []
            for _ in range(PERF_REPLAYS):
                g.replay()
                torch.cuda.synchronize()
                ts.append((int(ts_out.item()) - int(ts_in.item())) / 1000.0)
            del g
            med = float(np.median(ts))
            meds = [None] * gw
            dist.all_gather_object(meds, med, group=group)
            if rank == 0:
                meds_np = np.array(meds)
                perf[f"{m}/{w}"] = {
                    "per_call_median_us_rank0": med,
                    "per_call_median_us_max_rank": float(meds_np.max()),
                    "per_call_p95_us_max_rank": float(
                        np.percentile(
                            np.concatenate(
                                [np.full(1, x) for x in meds]), 95)),
                    "us": [round(x, 3) for x in meds_np],
                }
            dist.barrier(group)
    rec["perf"] = perf if rank == 0 else None

    _write(out, rec, rank=rank)
    dist.barrier(group)
    dist.destroy_process_group()


def _write(out: str, rec: dict, rank: int) -> None:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"rank{rank}.json").write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-g0", action="store_true")
    args = ap.parse_args()

    if args.no_g0:
        window = {"no_g0": True}
    else:
        window = check_window_g0()
        print(f"[G0] window OK: journal silent {window['journal_silence_s']:.0f}s",
              flush=True)
    out = args.out or f"/data/sm70_results/round5/ar_codec_debug"

    import torch.multiprocessing as mp
    port = 29631 + (int(time.time()) % 50)
    ctx = mp.get_context("spawn")
    # NOTE: Process.start() returns None — keep the Process objects.
    workers = [ctx.Process(target=worker, args=(r, port, out), daemon=False)
               for r in range(WORLD)]
    for w in workers:
        w.start()
    HANG_CEILING_S = 900.0
    t0 = time.monotonic()
    hung = False
    while any(w.is_alive() for w in workers):
        for w in workers:
            w.join(timeout=5.0)
        if time.monotonic() - t0 > HANG_CEILING_S:
            print("[unit] HANG past ceiling — 90s self-abort wait, then "
                  "TERM one by one", flush=True)
            time.sleep(90.0)
            hung = True
            break
    for w in workers:
        if w.is_alive():
            # SIGKILL, not SIGTERM: a rank that died can itself hang in
            # gloo teardown (busy-spin, no liveness timeout) and ignore
            # normal termination — 2026-10-15 run: 7 ranks alive at 100%
            # CPU long after printing their death tracebacks.
            w.kill()
            w.join(timeout=60)
    codes = [w.exitcode for w in workers]

    if hung or any(c not in (0, None) for c in codes):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "rank0.json").write_text(json.dumps(
            {"error": "hang or worker failure", "exit_codes": codes}))
        print(f"[unit] FAIL: hang={hung} codes={codes}")
        return 1

    rec = json.loads((Path(out) / "rank0.json").read_text())
    if "error" in rec:
        print(f"[unit] FAIL: {rec['error']}")
        return 2
    mm = {k: v for k, v in rec["model"].items() if v}
    all_ok = all(r["model_bitwise"] and r["bitwise_across_ranks"]
                 for r in mm.values())
    perf = rec.get("perf") or {}
    print("\n================ CODEC UNIT VERDICT ================")
    for k in sorted(mm, key=lambda s: (int(s.split('/')[0]),
                                       int(s.split('/')[1]))):
        r = mm[k]
        print(f"  model m={r['m']:3d} wire={r['wire']}: "
              f"{'PASS' if r['model_bitwise'] and r['bitwise_across_ranks'] else 'MISMATCH'}"
              f" n_mismatch={r['n_mismatch']} max_abs_err={r['max_abs_err']:.3e}")
    for w in WIRES:
        a = perf.get(f"1/{w}", {})
        b = perf.get(f"8/{w}", {})
        print(f"  perf wire={w}: m=1 {a.get('per_call_median_us_rank0', float('nan')):.1f} us"
              f" / m=8 {b.get('per_call_median_us_rank0', float('nan')):.1f} us"
              f" (rank0 medians; per-rank in json)")
    print(f"  VERDICT: {'PASS — all wire modes model-bit-exact; perf table '
          'collected' if all_ok else 'FAIL — mismatches: debug from the '
          'n_mismatch patterns above'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
