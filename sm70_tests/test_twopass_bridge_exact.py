"""bridge_kv: the bit-built e2m1 must be BIT-identical to the table it replaced.

Claim being tested: replacing the 8-float LUT with exact fp32 bit construction
removes a local-memory spill (STACK:32, STL:128/LDL:64) without changing any
value.  That claim is only safe if the two paths agree on every code, so compare
them element-for-element rather than statistically.

The table variant is the incumbent that already passed canary/recall/secret, so
bit-equality with it transfers that validation to the new path; no re-derivation
of nvfp4 page semantics is needed here (both variants decode the same bytes).

Also times both, because the whole point is the 9.3-9.9x gap to the bandwidth
floor.  Timing is min-of-N interleaved in one process (cross-process absolute
numbers are not comparable on this box).
"""
import torch

from vllm.v1.attention.ops.sm70_nvfp4_prefill_twopass import _mod

mod = _mod("nvfp4")
dev = "cuda"

H, NP, DD, SD = 2, 64, 128, 8          # 256 dims at 4 bit = 128 B/token; 16
# Offsets read off the kernel: K codes at kh*(NP*DD)+nn*DD, K scales at
# H*NP*DD + kh*(NP*SD)+nn*SD, and V is the same pair shifted by side_off.  So
# side_off must clear K codes *and* K scales, i.e. H*NP*(DD+SD); using H*NP*DD
# (the first guess) put V codes on top of the K scales and made the scale bytes
# vary with the code byte, which is what made the coverage check report 278
# distinct values instead of the 15 e2m1 magnitudes.
CODE_BYTES = H * NP * DD
SIDE = H * NP * (DD + SD)
PAGE = 4 * H * NP * (DD + SD)           # generous: only offsets matter here
DH = 256


def make(bn, kv_len, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    nblocks = (kv_len + NP - 1) // NP + 1
    kv = torch.randint(0, 256, (nblocks * PAGE,), generator=g, dtype=torch.int32,
                       device=dev).to(torch.uint8).contiguous()
    bt = torch.arange(nblocks, dtype=torch.int32, device=dev).unsqueeze(0)
    Kb = torch.zeros(bn, DH, dtype=torch.float16, device=dev)
    Vb = torch.zeros(bn, DH, dtype=torch.float16, device=dev)
    return kv, bt, Kb, Vb


def run(kv, bt, Kb, Vb, bn, kv_len, table):
    mod.bench_bridge(kv, bt, Kb, Vb, 0, 0, 0, bn, kv_len, PAGE, SIDE,
                     bt.size(1), H, NP, DD, SD, 1.035, 0.977, table)
    torch.cuda.synchronize()
    return Kb.clone(), Vb.clone()


print("=== 1. 位相同性（新 bit-built vs 原 table）===")
allok = True
for bn, kv_len, seed in ((64, 64, 1), (128, 200, 2), (256, 256, 3), (7, 33, 4),
                         (512, 1000, 5), (8192, 8192, 6)):
    kv, bt, _, _ = make(bn, kv_len, seed)
    ka, va = run(kv, bt, torch.zeros(bn, DH, dtype=torch.float16, device=dev),
                 torch.zeros(bn, DH, dtype=torch.float16, device=dev), bn, kv_len, 1)
    kb, vb = run(kv, bt, torch.zeros(bn, DH, dtype=torch.float16, device=dev),
                 torch.zeros(bn, DH, dtype=torch.float16, device=dev), bn, kv_len, 0)
    same = torch.equal(ka, kb) and torch.equal(va, vb)
    # every code value must actually have been exercised
    nz = int((ka != 0).sum().item())
    allok &= same
    print(f"  bn={bn:5} kv_len={kv_len:5}: K/V 位相同={same}  非零元素={nz:,}"
          f"  {'ok' if same and nz > 0 else '*** FAIL ***'}")
    del kv, bt, ka, va, kb, vb
    torch.cuda.empty_cache()
print("  位相同性:", "PASS" if allok else "FAIL")

print("\n=== 2. 覆盖全部 16 个 e2m1 码（含符号位）===")
# Fill only the CODE bytes with `code` and pin the group scales to the e4m3 byte
# that decodes to exactly 1.0 (s=0,e=7,m=0 = 0x38), so the decoded value is
# e2m1(nibble) * 1.0 * k_scale and the magnitude set is directly checkable.
E4M3_ONE = 0x38
CODE_BYTES = H * NP * DD
seen = set()
for code in range(256):
    kv, bt, _, _ = make(8, 8, 7)
    kv[:] = E4M3_ONE
    kv[:CODE_BYTES] = code
    kv[SIDE:SIDE + CODE_BYTES] = code
    k, v = run(kv, bt, torch.zeros(8, DH, dtype=torch.float16, device=dev),
               torch.zeros(8, DH, dtype=torch.float16, device=dev), 8, 8, 0)
    seen.update(x for x in torch.unique(k).tolist())
got = set(float(x) for x in seen)
# Kb is fp16, so value = round_fp16(e2m1 * 1.035); dividing back leaves ~1e-4 of
# representation slop, hence a tolerance rather than exact rounding (an earlier
# version rounded to 4 dp and declared every magnitude "missing").
def near(x, pool):
    return min(pool, key=lambda e: abs(e - x)) if pool else None


EXP = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0,
       -4.0, -6.0]
shape = set()
for x in got:
    e = near(x / 1.035, EXP)
    if e is not None and abs(x / 1.035 - e) < 2e-3:
        shape.add(e)
missing = set(EXP) - shape
print(f"  解出的唯一值 {len(got)} 个；覆盖的 e2m1 幅度/符号 {len(shape)}/15；"
      f"未覆盖: {sorted(missing) if missing else '无'}"
      f"  {'ok' if not missing else '*** 有码未覆盖 ***'}")

print("\n=== 3. 计时（同进程交错，min-of-8）===")
for bn in (4096, 8192):
    kv, bt, _, _ = make(bn, bn, 11)
    res = {0: [], 1: []}
    for rep in range(3):
        for table in ((1, 0) if rep % 2 == 0 else (0, 1)):
            K = torch.zeros(bn, DH, dtype=torch.float16, device=dev)
            V = torch.zeros(bn, DH, dtype=torch.float16, device=dev)
            for _ in range(2):
                mod.bench_bridge(kv, bt, K, V, 0, 0, 0, bn, bn, PAGE, SIDE,
                                 bt.size(1), H, NP, DD, SD, 1.035, 0.977, table)
            torch.cuda.synchronize()
            best = 1e30
            for _ in range(8):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                mod.bench_bridge(kv, bt, K, V, 0, 0, 0, bn, bn, PAGE, SIDE,
                                 bt.size(1), H, NP, DD, SD, 1.035, 0.977, table)
                b.record()
                torch.cuda.synchronize()
                best = min(best, a.elapsed_time(b) * 1e3)
            res[table].append(best)
            del K, V
    t_tab, t_bits = min(res[1]), min(res[0])
    # traffic: read codes+scale bytes, write 2*bn*DH fp16
    rd = bn * (2 * DD + 2 * SD)
    wr = bn * DH * 2 * 2
    tot = rd + wr
    for name, t in (("table(旧)", t_tab), ("bits(新)", t_bits)):
        print(f"  bn={bn:5} {name:9} {t:8.1f} us   {tot/t/1e3:6.0f} GB/s "
              f"({100*tot/t/1e3/900:4.1f}% HBM)   vs 旧 {100*(t/t_tab-1):+6.1f}%")
    del kv, bt
    torch.cuda.empty_cache()

print("\nVERDICT:", "PASS" if allok else "FAIL")
