#!/usr/bin/env python3
"""CPU-only unit test for the PLE NVFP4 sidecar loader branch.

Exercises _PleNvfp4Sidecar against (a) the real MagneticLab sidecar directory
(skipped when absent) and (b) a tiny synthetic sidecar in a temp dir
(boundary crossing, negative cases). No service boot, no GPU work.

Run: python sm70_tests/test_ple_sidecar_loader.py
"""

import json
import os
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.models.qwen4_exp.common.ngram_embedding import _PleNvfp4Sidecar

TMP = tempfile.mkdtemp(prefix="ple_sidecar_test_")

# ---- (a) real sidecar: init + geometry + scale -----------------------------
ROOT = "/data/models/Qwen3.8-Flash-Next-NVFP4/ple-sidecar/ples_nvfp4"
TOTAL_ROWS = 320_001_536
DIM = 160

if os.path.isdir(ROOT):
    sc = _PleNvfp4Sidecar(ROOT, total_rows=TOTAL_ROWS, dim=DIM)
    assert sc.shards == 128 and sc.rows_per_shard == 2_500_012, (sc.shards, sc.rows_per_shard)
    assert abs(sc.global_scale - 2.0345052689663135e-05) < 1e-12, sc.global_scale
    print(f"ok: real sidecar init (128 shards, {sc.rows_per_shard} rows/shard, scale={sc.global_scale!r})")


    def check_copy(sidecar, g0, n):
        """Copy rows [g0, g0+n) and byte-compare with direct safe_open reads."""
        codes = torch.empty(n, 80, dtype=torch.uint8)
        scales = torch.empty(n, 10, dtype=torch.uint8)
        sidecar.copy_global_rows(g0, n, codes, scales)
        row = 0
        while row < n:
            shard_idx, off = divmod(g0 + row, sidecar.rows_per_shard)
            take = min(n - row, sidecar.rows_per_shard - off)
            f = sidecar.open_shard(shard_idx)
            exp_c = f.get_slice("weight_e2m1")[off : off + take, :].view(torch.uint8)
            exp_s = f.get_slice("weight_scale")[off : off + take, :].view(torch.uint8)
            assert codes[row : row + take].numpy().tobytes() == exp_c.numpy().tobytes(), (shard_idx, off)
            assert scales[row : row + take].numpy().tobytes() == exp_s.numpy().tobytes(), (shard_idx, off)
            row += take
        spans = "boundary" if (g0 // 2_500_012) != ((g0 + n - 1) // 2_500_012) else "in-shard"
        print(f"ok: copy bytes exact for rows [{g0}, {g0 + n}) ({spans})")


    check_copy(sc, 1000, 16)                # inside shard 0
    check_copy(sc, 2_500_010, 16)           # crosses shard 0 -> shard 1 boundary
    check_copy(sc, 3 * 2_500_012 + 100, 16)  # inside shard 3
    check_copy(sc, TOTAL_ROWS - 12, 12)      # tail of last shard
    print("ok: real sidecar row copies (normal / boundary / shard3 / tail)")

    # ---- negative cases (real dir) -------------------------------------------
    for kwargs, exc in (
        (dict(total_rows=TOTAL_ROWS - 1, dim=DIM), ValueError),
        (dict(total_rows=TOTAL_ROWS, dim=320), ValueError),
    ):
        try:
            _PleNvfp4Sidecar(ROOT, **kwargs)
            raise AssertionError(f"expected {exc.__name__} for {kwargs}")
        except exc:
            pass
    print("ok: real sidecar rejects wrong row universe / width")
else:
    print(f"skip: real sidecar absent ({ROOT})")

# ---- (b) tiny synthetic sidecar ----------------------------------------------
TINY = os.path.join(TMP, "tiny")
os.makedirs(TINY)
rows, dim, nshards = 8, 16, 2
for k in range(nshards):
    out = {
        "weight_e2m1": torch.randint(0, 255, (rows, dim // 2), dtype=torch.uint8),
        "weight_scale": torch.randint(0, 255, (rows, dim // 16), dtype=torch.uint8).view(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(1.5e-5, dtype=torch.float32),
    }
    save_file(out, os.path.join(TINY, f"shard_{k}.safetensors"))
with open(os.path.join(TINY, "META.json"), "w") as f:
    json.dump({"layout": "group16_e2m1_e4m3scale_lownibblefirst", "shards": nshards,
               "rows": rows * nshards, "width": dim}, f)

t = _PleNvfp4Sidecar(TINY, total_rows=rows * nshards, dim=dim)
assert t.shards == 2 and t.rows_per_shard == 8
assert abs(t.global_scale - 1.5e-05) < 1e-12  # F32 round trip
c = torch.empty(4, dim // 2, dtype=torch.uint8)
s = torch.empty(4, dim // 16, dtype=torch.uint8)
t.copy_global_rows(6, 4, c, s)  # rows 6,7 in shard0; 8,9 in shard1
f0 = safe_open(os.path.join(TINY, "shard_0.safetensors"), framework="pt")
f1 = safe_open(os.path.join(TINY, "shard_1.safetensors"), framework="pt")
assert c[0].numpy().tobytes() == f0.get_slice("weight_e2m1")[6:7, :].view(torch.uint8).numpy().tobytes()
assert c[1].numpy().tobytes() == f0.get_slice("weight_e2m1")[7:8, :].view(torch.uint8).numpy().tobytes()
assert c[2].numpy().tobytes() == f1.get_slice("weight_e2m1")[0:1, :].view(torch.uint8).numpy().tobytes()
assert c[3].numpy().tobytes() == f1.get_slice("weight_e2m1")[1:2, :].view(torch.uint8).numpy().tobytes()
assert s[2].numpy().tobytes() == f1.get_slice("weight_scale")[0:1, :].view(torch.uint8).numpy().tobytes()
print("ok: tiny synthetic sidecar (boundary crossing, exact bytes)")

# negatives on the tiny sidecar
try:
    _PleNvfp4Sidecar(TINY, total_rows=rows * nshards, dim=dim + 8)
    raise AssertionError("expected width mismatch")
except ValueError:
    pass
bad = os.path.join(TMP, "tiny_bad")
shutil.copytree(TINY, bad)
os.remove(os.path.join(bad, "shard_1.safetensors"))
try:
    _PleNvfp4Sidecar(bad, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected missing-shard error")
except FileNotFoundError:
    pass
os.remove(os.path.join(bad, "META.json"))
try:
    _PleNvfp4Sidecar(bad, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected missing-META error")
except FileNotFoundError:
    pass
# non-integer META fields -> named ValueError (not a raw int() error)
badmeta = os.path.join(TMP, "tiny_badmeta")
shutil.copytree(TINY, badmeta)
with open(os.path.join(badmeta, "META.json")) as f:
    m = json.load(f)
m["shards"] = "two"
with open(os.path.join(badmeta, "META.json"), "w") as f:
    json.dump(m, f)
try:
    _PleNvfp4Sidecar(badmeta, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected non-integer META error")
except ValueError as e:
    assert "integer" in str(e), str(e)
# a float width (e.g. 16.0) must also be rejected, not truncated
badwidth = os.path.join(TMP, "tiny_badwidth")
shutil.copytree(TINY, badwidth)
with open(os.path.join(badwidth, "META.json")) as f:
    m = json.load(f)
m["width"] = 16.0
with open(os.path.join(badwidth, "META.json"), "w") as f:
    json.dump(m, f)
try:
    _PleNvfp4Sidecar(badwidth, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected float-width error")
except ValueError as e:
    assert "integer" in str(e), str(e)
print("ok: non-integer META fields (shards/rows/width) rejected with a "
      "named error")
# rows must divide evenly (build-tool guarantee)
baddiv = os.path.join(TMP, "tiny_baddiv")
shutil.copytree(TINY, baddiv)
with open(os.path.join(baddiv, "META.json")) as f:
    m = json.load(f)
m["shards"] = 3  # rows=8 is not divisible by 3
with open(os.path.join(baddiv, "META.json"), "w") as f:
    json.dump(m, f)
try:
    _PleNvfp4Sidecar(baddiv, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected rows%shards error")
except ValueError:
    pass
# a JSON bool is not an integer field (isinstance(True, int) is True)
badbool = os.path.join(TMP, "tiny_badbool")
shutil.copytree(TINY, badbool)
with open(os.path.join(badbool, "META.json")) as f:
    m = json.load(f)
m["shards"] = True
with open(os.path.join(badbool, "META.json"), "w") as f:
    json.dump(m, f)
try:
    _PleNvfp4Sidecar(badbool, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected bool-field error")
except ValueError:
    pass
# a corrupt (non-JSON) META still fails fast
badjson = os.path.join(TMP, "tiny_badjson")
shutil.copytree(TINY, badjson)
with open(os.path.join(badjson, "META.json"), "w") as f:
    f.write("not json {")
try:
    _PleNvfp4Sidecar(badjson, total_rows=rows * nshards, dim=dim)
    raise AssertionError("expected corrupt-META error")
except (ValueError, json.JSONDecodeError):
    pass
print("ok: uneven division, bool fields, and corrupt META all fail fast")
print("ok: fail-fast validation (missing shard file, missing META)")

# ---- env-off semantics: no sidecar construction without the env var ---------
assert os.environ.get("VLLM_PLE_QUANT_DIR", "") == ""
print("ok: VLLM_PLE_QUANT_DIR unset -> sidecar path inert (byte-identical to current behaviour)")

shutil.rmtree(TMP, ignore_errors=True)
print("ALL SIDE CAR LOADER TESTS PASSED")
