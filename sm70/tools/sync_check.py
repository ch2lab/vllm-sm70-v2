#!/usr/bin/env python
"""Post-sync checklist for the SM70 tree.

Every check here exists because a merge or a swap failed on it once:

  * the QSA page4 kernels and three Python modules were untracked, so the merge
    treated them as upstream deletions;
  * gdn_sm70_fwd.cu is JIT-compiled at runtime and is therefore absent from
    CMakeLists -- a build-reference-only audit called it dead and the merge
    deleted it, which broke the boot at kernel warmup;
  * the weight-cache daemon and the engine share a fingerprint protocol, so a
    one-sided swap cannot boot;
  * a stale include root or a dropped perf valve is invisible until the numbers
    move.

Usage:  sync_check.py <synced-tree> [<deployment-tree>] [<build-dir>]
Exit code is the number of failed checks.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def warn(name: str, detail: str) -> None:
    print(f"  [WARN] {name} — {detail}")
    WARNINGS.append(f"{name}: {detail}")


def git(tree: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), *args], capture_output=True, text=True
    ).stdout


def main() -> int:
    sync = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/data/src/vllm-sync")
    deploy = pathlib.Path(
        sys.argv[2] if len(sys.argv) > 2 else "/data/src/vllm-v2"
    )
    build = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "/data/sm70_build_sync")

    print(f"== SM70 sync checklist: {sync} vs {deploy} ==")

    # 1. nothing local may be untracked: two of the three boot failures were
    #    untracked production files that the merge read as deletions.
    untracked = [
        line[3:]
        for line in git(sync, "status", "--short").splitlines()
        if line.startswith("??")
    ]
    prod_untracked = [
        p for p in untracked if p.startswith(("vllm/", "csrc/")) and not p.endswith(".so")
    ]
    check("no untracked production file", not prod_untracked, ", ".join(prod_untracked[:5]))

    # 2. no merge leftovers.
    dirty = [
        line[3:]
        for line in git(sync, "status", "--short").splitlines()
        if line.startswith(("UU", "AA", "UD", "AU"))
    ]
    check("no unmerged path", not dirty, ", ".join(dirty[:5]))

    markers = subprocess.run(
        ["grep", "-rl", "^<<<<<<< ", str(sync)],
        capture_output=True,
        text=True,
    ).stdout
    check("no conflict marker", not markers.strip(), markers.splitlines()[:3])

    # 3. the deleted include root must be gone, and every path Python reads at
    #    runtime must exist (this is what a build-reference audit misses).
    stale = subprocess.run(
        ["grep", "-rn", "libtorch_stable", "--include=*.py", "--include=*.sh",
         "--include=*.cu", "--include=*.h", "--include=*.cuh", str(sync)],
        capture_output=True,
        text=True,
    ).stdout
    # A comment or docstring is harmless; a path built by concatenation
    # ('"csrc" / "libtorch_stable" / ...') is a runtime lookup and is not.
    # Only a lookup breaks a boot: a path segment ("libtorch_stable" as its own
    # quoted token, or "csrc" / "libtorch_stable") or a file access on the line.
    # A mention inside a message or a docstring is worth reviewing, not failing.
    stale_lines, text_mentions = [], []
    for ln in stale.splitlines():
        if "libtorch_stable" not in ln:
            continue
        content = ln.split(":", 2)[2] if ln.count(":") >= 2 else ln
        if re.search(r'"libtorch_stable"', content) or re.search(
            r"(open\(|Path\(|sources=|os\.path|load\(|_repo_root)", content
        ):
            stale_lines.append(ln)
        elif not re.search(
            r"^\s*(#|//)|``|\"\"\"|docstring|is included|Faithful model|codec functions|copied unchanged",
            content,
        ):
            text_mentions.append(ln)
    check("no live csrc/libtorch_stable reference", not stale_lines,
          f"{len(stale_lines)} line(s)")
    if text_mentions:
        warn("stale path in prose (not a lookup)",
             f"{len(text_mentions)} line(s): "
             + ", ".join(m.split(':')[0].split('/')[-1] for m in text_mentions[:3]))

    missing_jit: list[str] = []
    for py in sync.glob("vllm/**/*.py"):
        text = py.read_text(errors="ignore")
        for match in re.finditer(r'"(csrc/[^"]+\.(?:cu|cuh|h))"', text):
            if not (sync / match.group(1)).exists():
                missing_jit.append(f"{py.relative_to(sync)} -> {match.group(1)}")
    check("every csrc source named in Python exists", not missing_jit,
          "; ".join(missing_jit[:4]))

    # 4. build parity: same object set modulo upstream's own additions/removals.
    def objects(build_dir: pathlib.Path) -> set[str]:
        out = subprocess.run(
            ["ninja", "-C", str(build_dir), "-t", "targets", "all"],
            capture_output=True, text=True,
        ).stdout
        return {ln.split("/")[-1] for ln in out.splitlines() if ".o:" in ln}

    if build.exists() and (deploy / "build-sm70").exists():
        sync_obj, deploy_obj = objects(build), objects(deploy / "build-sm70")
        missing = deploy_obj - sync_obj
        extra = sync_obj - deploy_obj
        print(f"       objects: sync {len(sync_obj)} / deploy {len(deploy_obj)}")
        warn("object-set delta", f"missing {len(missing)}, extra {len(extra)}")
        for name in sorted(missing):
            print(f"         missing: {name}")

    # 5. the perf valves the serve script exports must still be reachable.
    serve = pathlib.Path("/root/run/run_sm70_serve_sync.sh")
    if serve.exists():
        names = sorted(set(re.findall(
            r"export (VLLM_[A-Z0-9_]+|SM70_[A-Z0-9_]+)", serve.read_text()
        )))
        corpus = "\n".join(
            p.read_text(errors="ignore")
            for p in list(sync.glob("vllm/**/*.py")) + list(sync.glob("csrc/**/*.cu"))
        )
        gone = [n for n in names if n not in corpus]
        check(f"all {len(names)} perf valves reachable", not gone, ", ".join(gone))

    # 6. op-name parity between the built extension and the serving one.
    sync_so = sync / "vllm/_C_stable_libtorch.abi3.so"
    deploy_so = deploy / "vllm/_C_stable_libtorch.abi3.so"
    if sync_so.exists() and deploy_so.exists():
        def ops(path: pathlib.Path) -> set[str]:
            out = subprocess.run(["strings", "-a", str(path)],
                                 capture_output=True, text=True).stdout
            return {ln for ln in out.splitlines()
                    if re.fullmatch(r"(sm70|qsa)_[a-z0-9_]+", ln)}
        s, d = ops(sync_so), ops(deploy_so)
        check("sm70/qsa op-name parity", s == d,
              f"only-sync {sorted(s - d)[:4]} only-deploy {sorted(d - s)[:4]}")

    # 7. op registrations that lived in the deleted central bindings. A kernel
    #    can survive a merge while its registration does not, and the failure
    #    only appears at kernel warmup ("'_C' object has no attribute ...").
    tb = git(deploy, "show", "HEAD:csrc/libtorch_stable/torch_bindings.cpp")
    if tb:
        registered = set(re.findall(r'ops\.(?:impl|def)\(\s*"([A-Za-z0-9_]+)', tb))
        live = set()
        for src in list(sync.glob("csrc/**/*.cu")) + list(sync.glob("csrc/**/*.cpp")):
            text = src.read_text(errors="ignore")
            live.update(re.findall(r'ops\.(?:impl|def)\(\s*"([A-Za-z0-9_]+)', text))
        # Upstream may have deliberately retired an op (wrapper and all) as dead
        # code; that is a warning with a recorded reason, not a failure.
        KNOWN_RETIRED = {
            "masked_situ_and_mul": "upstream removed the wrapper as dead code; "
                                   "no Python caller in either tree",
        }
        gone = sorted(registered - live)
        broken = [g for g in gone if g not in KNOWN_RETIRED]
        check(f"all {len(registered)} bindings from the deleted file registered",
              not broken, ", ".join(broken))
        for name in gone:
            if name in KNOWN_RETIRED:
                warn(f"retired op {name}", KNOWN_RETIRED[name])

    # 7. dependency drift: informational, but it decides the venv.
    req = sync / "requirements/cuda.txt"
    if req.exists() and (deploy / "requirements/cuda.txt").exists():
        def pins(path: pathlib.Path) -> dict[str, str]:
            return dict(re.findall(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==([^\s]+)",
                                   path.read_text(), re.M))
        a, b = pins(deploy / "requirements/cuda.txt"), pins(req)
        drift = {k: (a.get(k), b[k]) for k in b if a.get(k) not in (None, b[k])}
        if drift:
            warn("dependency drift", "; ".join(
                f"{k}: {v[0]} -> {v[1]}" for k, v in sorted(drift.items())))
        else:
            print("  [OK ] no pinned-dependency drift")

    print(f"\n== {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s) ==")
    for f in FAILURES:
        print(f"  ! {f}")
    return len(FAILURES)


if __name__ == "__main__":
    sys.exit(main())
