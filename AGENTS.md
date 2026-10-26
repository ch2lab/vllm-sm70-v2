# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.
- User-visible changes (features, CLI args, config fields, env vars, models,
  hardware, defaults, deprecations) must update the rendered docs in the same PR;
  follow [Documenting User-Visible Changes](docs/contributing/README.md#documenting-user-visible-changes).
- Before opening a PR (drafts included) or requesting re-review, run the [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill and address its findings.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# Start with precompiled artifacts for an editable install:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md) to
configure and perform incremental builds.

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that actually reproduces it? Fixes to user-visible
  behavior need a test through a real entry point (`LLM`, OpenAI server); see
  [`tests/AGENTS.md`](tests/AGENTS.md).
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## 3. SM70 Hardware-Ceiling Optimization Discipline

Scope: 8×V100 SXM2 TP8, Qwen3.8-Flash-Next-NVFP4 (全 nvfp4 KV), decode + prefill
逼近硬件天花板。目标 = 逼近天花板, 不是 3% 增量; 逐环节判 HW-BW / HW-latency / 算法 / launch。

Workflow (mandatory order, 不得跳步):
1. **H1** — 下界 + gap 分解表 (离线 bench + 配置算术, 无 profiler) 定序。
2. 单变量 → **valve (可回滚)** → 同一方法测量 → 全矩阵 → 归档 → 保留/回滚。

Measurement (唯一 GO/NO-GO): serve 走 sm70-serve.service; **65K ruler salt 9502 +
qsa_conc_tput jsonl**。离线 bench 仅用于诊断/下界, 不得作 GO/NO-GO。

Policy v2 (2026-10-19, user-directed: 解除全部限制, 持续迭代优化):
- **允许**: 配置 / 并行布局 (TP·DP·PP) / 模型与量化 / 稀疏预算 / **profiler (nsys·ncu·torch profiler)** / 内核重写 /
  多流与微批 / 投机解码 (MTP). 旧的"禁改配置/禁改布局/禁碰模型"全部作废.
- **判据**: 由 **65K ruler (salt 9502) + 内容门 + 0 新 Xid** 决定去留; **不再要求单次 ≥3%** —
  小步快跑, 但每次改动必须: 备份 → 过门 → 留回滚点; 不许把未验证的实验留在生产态.
- **保留的作业纪律 (安全/正确性, 不是性能限制)**:
  ① 改动前备份 (脚本/源码/.so)   ② 每次 boot 前 `ram_guard.sh`   ③ **先证生效** (canary/日志/sha) 再读数
  ④ 基线必须**同期**测 (记忆锚会漂 ±2%)   ⑤ 补丁脚本"先断言全部锚点、再一次性写回"
  ⑥ 跨进程绝对值不可比 (时钟漂移), 消融/相位必须同进程交错取 min
  ⑦ 实验性改动可回滚 (留备份); 已知 bug 仍**原地修复**并过 correctness.
  ⑧ **铁律: 禁止用 HTTP /health 探测服务状态**。一律 `journalctl -u sm70-serve` **实时读日志**判定:
     成功 = 出现 `Application startup complete`; 失败 = 出现 `Traceback` / `...Error:` 行 / unit failed.
     **失败必须打印原始日志尾部**(不要只看包装后的 RuntimeError, 要看到 AttributeError/RuntimeError 的真行).
     统一走 `/root/run/serve_wait.sh [timeout] [unit]` (成功 exit 0, 失败 exit 1 并打印 RAW TAIL, 超时 exit 2).
  ⑨ **sm70-serve 当前是测试台, 不是生产**(用户指令 2026-10-19): 无人在线 ⇒ **不做"立刻恢复"、不拿 health 当门**。
     唯一要维护的是**基线快照**: 基线脚本 `/root/run/run_sm70_serve_ipc.sh`(备份 `/tmp/opencode/serve_pre_*.sh`) +
     daemon 脚本(8192/TS8) + 基线性能锚(**2026-10-24 复核; 当前一代 = flat AR + FP4-Local-NVFP4**, salt 9502/9503) =
     prefill 65K hdr **8687** (8687/8705; hier 8684/8691) / decode1 65K **109.6 tps**(MTP k=1, 中位 n=5, 98.0–112.0;
     hier 100.9; **无 MTP ≈80**) / conc8 65K **404.7 agg**(hier 355.2) / canary **9/9** / Xid 0.
     ⇒ 试验可**任意叠加、任意重启**; 失败只记**日志真行**, 立刻下一个候选。
     **旧锚已作废**(勿再引用): `5875–5923 / 60.9 / ≈295`(旧模型) 与 `6055–6066 / 66.0–66.9 / 312–317`(旧 wire 时代) —
     权威出处 `sm70/19_two_layer_ar_protocol.md` §8/§10。
     只有两种情况需要恢复基线并过 canary: ①要做"干净基线对照"的一次测量 ②要交给人/对外服务。
     注意: 服务端配置改动(如 chunk/EP/HC_TP)会让 ipc 缓存键不匹配 ⇒ loader 会**自动回落磁盘加载**(已实测) ⇒ 不必重建 daemon。
  ⑩ **纪律: 脚本一律后台跑, 定时短命令查** (用户指令 2026-10-25, 违反过两次):
     长任务必须 `setsid nohup <script> ><log> 2>&1 &` **脱离会话**启动 —— 前台长跑时工具调用一旦被中断,
     整个进程组被带走(实测两次把 TP4 双实例杀成全卡 0 MiB)。等待用 goal_wait, 不要在同一命令里 `sleep` 长等。
     **判进度只认 GPU 显存 + 进程年龄**, 不认日志行(同一 log 跨 run 追加, "A ready" 可能是上一代残留)。
     serve/daemon 顺序: `sm70-serve` 带 `Requires=sm70-daemon`, serve 启动事务排队时**不要** restart daemon
     (会取消 daemon 的启动任务, 实测 7 分钟一次重载循环) ⇒ 先 stop serve 等 cgroup 排空 → start daemon →
     等 READY(锚定 unit 启动时间戳; 注意 READY 行比 systemd `ActiveEnterTimestamp` **早约 2 s**, 用 `@(EPOCH-60)`)
     → `systemctl start serve`(不要再用 `serve_restart_safe.sh`, 它开头又 stop 一次)。daemon 满载约需 7 分钟。
- **当前 prefill 地图 (2026-10-19, `--enforce-eager` + opattr 全 coverage, 总量 11.4 s ≈ 墙 11.08 s)**:
  MoE ≈25% (在 `mlp_hc` 区内) · QSA indexer/选择 ≈15% · **`in` (PLE/embed/norm) ≈11.5%** ·
  QSA `pre` (planner/KV) ≈10% · GDN attn ≈10% · HC down/up ≈9% (地板 154 ms/rank ⇒ 6.5× off) ·
  QSA 稀疏本体仅 ≈1.3%.  ⇒ 靶子顺序由这张表决定, 不再沿用旧份额.

Topology (nvidia-smi topo -m 实测, 不得再猜): **单一 8-GPU NVLink 网** (每卡 6 link / 4 邻居,
NV2≈51.6 / NV1=25.8 GB/s), 无 NVSwitch, **无孤立岛**; 0-3/4-7 只是 NUMA 分组, 4 条 NVLink
跨 NUMA (0↔7,1↔6,2↔5,3↔4) → 全 8 卡 AR ring 可纯 NVLink (不经 PCIe)。HBM 900 GB/s/rank。

Records: durable conclusions → **append to `sm70/03_history.md`** (the loadable
source), NOT a new scattered .md; bench scripts `sm70_tests/bench_*.py`; raw data
`/data/sm70_results/<iterN>/<exp>/` (never /tmp).

Anchors(**旧时代, 已作废, 勿引用**): decode1 17.2 ms/步 (58 tps); conc8 293–297 tps ⇒ 当前锚见上方 §3 ⑨。
dense streaming 下界 1.28 ms/rank/步(仍成立)。
已证伪: 发射链路 (eager 4.5× 慢)。已归档: C1 Triton GEMV 子集。

---

## 3a. SM70 Reference — single source: `sm70/` (read `00` first, then as needed)

Do NOT re-derive deployment/hardware/history from memory — it lives in the
loadable `sm70/` docs (this section is only an index):
- **sm70/DISCIPLINE.md** — READ FIRST before any GPU/measurement work: the one-page
  hard rules (anchors, no-HTTP probing, restart order, candidate state machine,
  `/root/run/measure_gate.sh`). Details live in `04_standards.md`.
- **sm70/RUNBOOK.md** — THE standard experiment shape (single entry: `measure_gate.sh`
  + `wait_job.sh` + `svc.sh`; failure-class→enforcement table). Use it for every measurement.
- **sm70/DEAD_PATHS.md** — dead-path registry (what was disproven, what to DELETE,
  what is fix-track). Check before resurrecting any valve or kernel.
- **sm70/00_machine_state.md** — resident processes, VRAM budget
  (32.7 GB/card: 18 preload + ~9.7 serve + ~5 free), what can/can't be killed.
- **sm70/01_deployment.md** — preload→serve (8002→8001 `ipc_cache`), ports, prod
  perf valves, serve flags.
- **sm70/02_model_hardware.md** — model (48L: 12 QSA + 36 GDN + MoE) + V100 SXM2
  (single 8-GPU NVLink mesh, no NVSwitch, no BF16 TC).
- **sm70/03_history.md** — optimization history, BOTH eras: deployed wins,
  disproven/archived, baselines.
- **sm70/04_standards.md** — measurement (65K ruler salt 9502), keep bar, iteration
  discipline, file conventions.
- **sm70/05_perf_route.md** — Rounds 1–4 detailed route (sourced mechanism + numbers).
- **sm70/06_ple_sidecar.md** — PLE NVFP4 sidecar contract (frozen-table, scale
  adopt-or-raise, reload-safe path).
- **sm70/07_h1_gap.md** — H1 gap table: per-stage wall (measured) vs HW ceiling class
  (config), corrected AR=NVLink; GPU/offline-bench cells marked TBD.
- **sm70/08_fp4_local_nvfp4_adopt.md** — 采纳新模型 FP4-Local-NVFP4 设计
  (PLE NVFP4 直通 + 视觉), 状态/WBS/回滚阀见文档; 2026-10-18 设计完成待执行.

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
- **Rust frontend (`rust/`)**: [`rust/AGENTS.md`](rust/AGENTS.md)
  — Crate ownership, design principles, Python parity, and testing conventions.
