# SM70 Standards & Method

> **纪律清单的唯一权威在 [`sm70/DISCIPLINE.md`](DISCIPLINE.md)**（一页版：锚点 / 禁 HTTP 探活 /
> 重启顺序 / 候选状态机 / measure_gate）。本文件保留**细则与历史案例**（测量细节、单位陷阱、
> 证据分级、事故复盘）；两者冲突时以 DISCIPLINE.md 为准，并应在本文件补一段案例说明为什么。

## Measurement (the ONLY GO/NO-GO method)

**口径（user ruling 2026-10-18，已上线）：prefill/decode 速度取服务端响应体数据 —— `usage` 的
token 计数 + `metrics` 的计时。不用客户端本地统计，也不用 `X-Request-Perf` header。**

- **`--enable-per-request-metrics` 已固化进 `/root/run/run_sm70_serve_ipc.sh`（2026-10-18）。
  纯报数**：应用后 65K salt 9502 → prefill 5229/5261/5260、decode 59.0、Xid 216 = 基线（Δ0）。
  响应体与流式末块携带 `metrics`（`PerRequestMetrics`）；`n>1` 时该字段被抑制。
- 定义（2026-10-18 活服务实测自洽，2690-token prompt）：
  - `prefill_tps = (usage.prompt_tokens − usage.prompt_tokens_details.cached_tokens) / (metrics.time_to_first_token_ms/1000)`
    —— **有效重算量**（`USAGE_DETAILS.md` §5.4 计费口径）；缓存命中部分不是本次重算。
  - `decode_tps  = 1000 / metrics.mean_itl_ms`
    （`completion_tokens/(metrics.generation_time_ms/1000)` 会略偏：decode 区间自首 token 起算、不含它。）
  - ⚠️ **`metrics.tokens_per_second` 不是 decode 速率**：它 = `completion_tokens/(last_token_ts − scheduled_ts)`，
    **把 prefill 算进去了**（实测 32 tok /(589.31+521.02)ms = **28.82**，而 decode 实为 **59.50**）。不要用它。
  - 防御读法（`USAGE_DETAILS.md` §5.1/§3.2）：键缺失 ≡ null ≡ 0；流式只认最后一个 `choices:[]` chunk，
    不跨 chunk 累加（中间 chunk 是累计值）。
- 辅助交叉核对（**不作准**）：`X-Request-Perf` header、服务日志行、`baseline.py` ruler 的 hdr 列。
  实测一致：metrics.decode 59.50 ↔ header 62 ↔ 既有锚点 59.3–59.4；metrics.ttft 589.31 ↔ header prefill ms 608.7。
- **本地墙钟统计不作 KPI**（`sum(prompt_tokens)/max(ttft)`、`t_end−ttft` 之类）：同一 65K conc1
  同一时刻，客户端口径 4378 vs 服务端口径 5263（差 20%）；且缓存命中轮会给出荒谬值
  （旧 decode-mode 轮 `prefill_agg≈100000 tok/s`，实为 `cached_tokens=65072` 被当成重算量）。
- **验证（2026-10-18，活服务，1490-token prompt）**：冷启动 `cached=0` → 旧口径 = 新口径
  = **3720.3** tok/s（同值 ⇒ 规则对既有冷跑记录向后兼容）；同前缀第 2 次 `cached=784` →
  旧口径 **7594.3**（把 1490 全算重算量，虚高 2×）vs 新口径 **3598.4**（真值）。
  ⇒ **旧口径在缓存命中时虚高**；10-16 jsonl 里 `cached=65072` 的行报 67044/69640 tok/s 即此类。
  **历史"6000+"这类数字的部分来源就是缓存虚高，不是被丢掉的优化。**
- **未解释的 17–20% 波动（2026-10-18 实测）**：同脚本同 65K conc1，seed 952101000 → 服务端
  header 4384；seed 952600000 → 5157/5179（客户端同向）。候选原因：提示词内容影响 QSA 稀疏
  选择密度 / 引擎·GPU 残留状态。⇒ **单次单提示词测量不作结论**：固定 salt 序列 + 多次重复。
- 65K ruler 仍为 **salt 9502**，`qsa_conc_tput.py`：
  - `--mode prefill --len N --conc 1 --rounds 2`（`max_tokens=1`；guard: 每请求 cached=0）
  - `--mode decode --len 65536`（warm 轮填 APC、timed 轮命中）→ `decode_agg_tps`
- **Fresh, distinct salt per (len, run)** — reusing a salt re-hits the prefix cache
  (cached>0 = INVALID round).
- Results → `/data/sm70_results/<iterN>/<exp>/`. **Never /tmp** (volatile).

### 声明纪律（user ruling 2026-10-19）：每条机制必须带标签
任何进入表格/结论的机制，前面必须标 **`[实测]` / `[文档·待复核]` / `[推断]`**：
- **`[实测]`**：本 session、同一装置同一口径、配对或重复测过 —— **写清 n**
- **`[文档·待复核]`**：引自历史记录 / 异配置 / 已归档 trace ⇒ 用前走"三步核"（核后端、核内核名在本树存在、注明测得配置），
  且**不得据此做 GO/NO-GO**
- **`[推断]`**：读码 / 建模 / 假设 ⇒ **不得进"证据"栏**，只能进"待验假设"

**硬规则**
1. **不同分母 / 不同配置的份额不得混算**（实例：把 `AGENTS.md` 那句 MoE≈25%/GDN≈10% 与今天配置的账混用 ⇒ 得出错误的效率排序 ✗）
2. **n=1 必须写明**（单次 boot 的 Δ 不能与多轮重复的锚点等量齐观）
3. 引用别家数字（1cat/Strata）必须写明"**在其形状上**"；迁移性未证前不得当收益承诺
4. 自查义务：发现自己此前的结论与**本树自己的台账**矛盾时，先按矛盾处理（实例：HC_TP 的"每层多一次 AR"与
   `07_h1_gap.md` 的 `6 AR/层（含 4 个 HC AR）` 矛盾 ⇒ 该机制降级为"未归因"）

### ⛔ 单位陷阱（同一个坑踩了两次：2026-10-18、2026-10-19）
`qsa_conc_tput --mode prefill` 报的 `hdr_prefill` / `prefill_agg` 是 **tok/s**，**不是毫秒** ✗。
**65K prefill 墙（秒）= 65536 / hdr_prefill**（≈ **11.1 s** @5900 tok/s；冷首请求 `ttft_max` ≈11.4–11.5 s 相符）。
把 tok/s 当秒，会让份额、天花板、"还剩几倍"**全部错**（我据此得出过 12.8% 峰值，真值 **6.8%**）。

### 运维三条（2026-10-19 实测，避免重复浪费）
- `systemctl **start**` 对**已在运行**的 unit 是**空操作** ⇒ 换配置一律 `restart`（否则 serve 拿到旧 daemon 的键）
- `serve_wait.sh` 的 daemon 分支曾**假超时**（READY 标记**早于** `restart` 返回时刻 ⇒ 永远扫不到窗口内）⇒
  已改为按主进程启动时刻扫；实测 **0.33 s** 判定（此前空等过 1.5 h）
- `pkill -f "<模式>"` 会**匹配到自己的命令行**（我因此把自己 SIGTERM 掉一次）⇒ 一律用括号法：`pkill -f "daemon_[b]atch"`

### 长上下文质量门（必须走 `23_quality_gate_standard.md`）

任何会改变"模型能看见多少上下文"的改动（有界/窗口注意力、稀疏选择、KV 精度/压缩、分块、内核替换）
都必须过三级门：**L0 数值门**（离线对拍 `rel ≤5e-4`，且**同 shape 重复 ≥3 次**，长度矩阵必须含
**非 16 对齐**值如 8040/9000 —— 竞态只在那里暴露）→ **L1 行为门**（首 token top-5 logprob 两臂对比，
top-1 一致率 ≥90%、`|ΔNLL| ≤0.05`）→ **L2 任务门**（RULER/LongBench v2 子集，Δ ≤2 分）。
铁律：只比 Δ、对照必须通过、同臂重复必须逐位一致；`INVALID`（复读/空答）>20% ⇒ 测法失效，不是模型不行。
全套规格与执行清单见 **`sm70/23_quality_gate_standard.md`**。

## Offline benches
- GPU benches run under `/data/vllm-dev`, **quiet window first** (journal/service
  silent, GPU idle; resident daemon/preload VRAM is fine).
- **Contamination canary is mandatory (2026-10-18 lesson)**: this box has multiple
  logged-in users, and another session's GPU job silently inflated a whole sweep
  ~2x (the in-run mem-eff reference itself degraded 2.3x) while `nvidia-smi` still
  read 0% util at the sample points. **Before trusting any offline GPU number, the
  run must measure a reference op whose clean value is known and show it within
  ~1.25x.** `sm70_tests/bench_fmha_prefill_t12_v2.py` implements this as a canary
  that aborts with exit 2. A bench without a canary is not evidence.
- Offline = **diagnosis / lower-bound only, never GO/NO-GO**. Beware: production mixes
  KV dtypes (fp16 QSA / nvfp4 other) + specific kernels — a bf16/fp16 micro-bench may
  NOT match production.

## Iteration discipline

### 不可放弃原则（user ruling 2026-10-18；最高优先，压倒其它裁决便利）

**默认立场：任何可能带来优化的候选都必须保留在册，直到有"有据的证据"证明其不可行。**
"有据的证据" = 可复现、可复核、且**针对该方案本身（而非某个具体实现）** 的证明，仅限：
- 该方案的**理论上界**已低于目标（附算式：如某数学重构的 FLOP/字节下界 > 目标墙）；
- 该方案**自由度已穷尽**（逐一列出所有自由度 + 各自实测数字，且全部低于门槛）；
- 该方案触犯**硬停**（改配置 / 改布局 / 不可接受的精度），且不存在等价绕法。

**以下理由一律不构成"不可行"，不得据此关闭候选**：
- 未过某门槛 —— 尤其门槛是**借自另一个 effort** 的 ⇒ 转修复轨 / 重判门槛；
- 质量或精度失败（`correctness.py` FAIL、rel-L2 超限、precision collapse）⇒ **修复轨**；
- 归因所依赖的前提/机制**后来被推翻**（例：拓扑、测量口径）⇒ **先复核再判**；
- "回退了 / no-diff" **但没有定位**到算子或代码路径 ⇒ 未定位即未关闭；
- 工作量、复杂度、"已经试过几轮" ⇒ 只是排期问题，不是不可行；
- 单次、单配置、单 salt、单形状的测量 ⇒ 不足以判死。

**候选状态机（每个候选必须处于其一，并且必须在册）**：
`open → (banked | fix-track → banked | blocked-by-evidence(附证据) | dropped-by-user)`
- **"排队 / behind gate" 不是终态**：必须带排期，否则视为 `open` 且排在队首。
- **周期性复核**（每次主线换手时）：重审所有 `blocked-by-evidence` 与 `dropped` 的**依据是否仍成立**；
  依据失效 ⇒ 自动回到 `open`。
- 台账（live ledger）维护在 `03_history.md` 的 "OPEN-CANDIDATE LEDGER" 节。

### 候选裁决：投产安全轨 ⊥ 修复轨（user ruling 2026-10-18，替代 "correctness FAIL ⇒ 丢弃"）

- **两条独立轨，不许混淆**：
  1. **投产安全轨（安全动作，非裁决）**：只要候选会产出**错误内容**（`correctness.py` FAIL）
     或引入 Xid，就**不得留在生产** —— 把 **valve 关掉**。`git stash`/checkout 仍禁止。
     关阀 = 保护线上正确性，**不是**对候选的结论。
  2. **修复轨（FIX TRACK，默认走向）**：候选一旦显示**真实速度收益**（某轴 ≥3%，或显著），
     就**直接进入修复轨**并立档：收益数字 + 失败判据 + 复现命令 + 归因方向 → 定位算子/代码路径
     → **原地修复** → `correctness.py` ∧ 65K ruler ∧ 0 新 Xid 复验 → 再开阀复测。
     **"速度上去了但质量不过" 永远不等于丢弃 —— 那是待修的 bug，不是被否的优化。**
- **终止候选的仅三种情形**：① 修复 N 轮后收益被吃平（须附算术）；② 修复必须改配置/布局（硬停）；
  ③ 用户裁决放弃。**因为质量失败就把候选丢进废纸篓 = 违规。**
- **"已关闭 / 不复测" 的适用范围（本规则的关键限制）**：只对**有可复现算术或实验**证明"真耗尽 /
  真低于门槛 / 真无端到端价值"的方向成立。凡因下列原因关闭的，**一律可重开（进修复轨）**：
  ① **未过某个门槛** —— 特别是该门槛是**借自另一个 effort** 的（例：FMHA/GDN 被 P3-spec 预注册的
  2× 门挡住，而通用采纳条只有 ≥3%/轴）；② **质量/精度失败**（例：AR wire e4m3/nvfp4 = precision
  collapse）；③ **归因所依赖的前提后来被推翻**（例：AR relay/sys2hop 的 SYS-leg 分级基于已废弃的
  "2-岛"模型，真实为全 8 卡 NVLink mesh）。重开时先复核原裁决依据是否仍然成立。

- **按优化单元提供代码与记录（user ruling 2026-10-18）**：每个优化单元必须
  ① **一个可回滚的阀/env**（默认 = 现状；修复类若无阀则该 commit 只含该修复）
  ② **独立 commit**（禁止把两个单元混入一个 commit / 一个阀）
  ③ **原始数据归档** `/data/sm70_results/<unit>/`
  ④ **在 `sm70/05_perf_route.md` 的"优化单元台账"登记**（单元 / 阀 / 触点文件 / commit /
  证据锚 / 轴 / 状态：banked|doing|reverted|closed|fix-track）。
  目的：后续能**按单元区分优化路径**（谁在跑、谁被回滚、谁在修复轨、证据在哪）。
- **保留判据（bank 条件）**：某轴 ≥3% ∧ 内容门 PASS ∧ 0 新 Xid。
- **被重叠的集合通信：只能用"服务内边际/关键路径"口径（2026-10-18 教训）**。孤立单次成本
  （如 codec 单测在空闲机、单 AR、图内计时）与**服务内关键路径**成本可以**符号相反**：
  本例 fp16 wire 孤立 **22.5 µs（比 fp32 的 45.0 更快）**，但服务内边际 **27.0 vs 23.9 µs（+13%）**、
  conc8/16/32 吞吐 **−2.6/−3.1/−4.0%** —— 因为服务里 AR 与计算**重叠**，孤立计时看不到重叠被打断的代价。
  **孤立单测只能用于**：① codec 正确性/逐位等价 ② 定位实现病态（如 e4m3 的 2.3× 串行化）。
  **判 GO/NO-GO 一律用服务内边际（AR 探针斜率）或吞吐曲线。**
- **历史分解会过期：用前必须按当前阀门状态复核（2026-10-18 教训）**。文档里形如"AR 占 15% /
  6 ARs per layer / 某组 X%"的账，都是在**某个阀门状态下**测的；阀门一变（例：`VLLM_SM70_HC_TP`
  默认 False ⇒ hc 的 down/inject 是 `ReplicatedLinear` ⇒ **那 4 个 HC AR 根本不存在**），
  旧账就作废。**使用任何份额之前**：先核 serve 脚本 + `/proc/<pid>/environ` + 相关 env 默认值
  （本轮 30 秒就查出"288 ARs/步"是 `HC_TP=True` 时代的账）。**不核就用 = 按过期前提做工程。**
- **份额/内核名三步核（2026-10-18 升级，第三次踩坑后）**：本 session 有 **三次**因引用别配置的份额而误导
  工程——① `07` 整张 P7 X-ray 表（异配置 + 含今天不存在的 `qsa_page4::` 内核）；② decode 墙的
  "12 µs/内核"（重构值，实测 4.6 µs，高估 2.6×）；③ **FMHA**：把 `fmha_cutlassF`（flashinfer 的 FMHA，
  = torch SDPA `EFFICIENT_ATTENTION`，属 P7/1Cat 配置）当成"serving 的 #1 prefill 内核"，为它投入了
  三轮内核工程，而**今天的 QSA 注意力是 Flash-V100 的 `flash_attention_*` 稀疏链，二者毫无关系**。
  ⇒ **任何"份额 / 内核名"用于工程前必须三步核**：
  ① **核后端选择**（当前 serve 阀门 + 路由 env/控制文件）；② **核该内核名在本树存在**
  ——源码 `grep`，或对运行时 `.so` 做 `nm -D --defined-only | grep`；
  ③ **注明它是在哪个配置下测的**。缺任一步 = 按过期前提做工程。
- **内容门必须覆盖候选的实际生效包络（2026-10-18 教训，代价：差点 bank 一个错内核的 +19%）**：
  若候选只在某形状区间生效（例：chunked GDN 仅在 `T ≥ 4096` 才被调用），则**必须在那个区间里验内容**。
  `sm70_tests/correctness.py` 的提示词全部 ≤2k ⇒ 它**测不到** `T≥4096`，短门全过 ≠ 内核正确。
  长上下文内容门的最小做法：≥4096 token 的提示词 + needle 检索（答案可核）+ 与现役路径同输入对照。

### 其余（不变）
- 单变量 → valve（可回滚、默认关）→ 全矩阵（65K ruler）→ 归档 → 保留 / 转修复轨。
- **Hard stops:** no config changes · no layout changes (keep TP8) · 0 new Xid ·
  never leave production in an experimental config.

- **No profiler** — attribution via offline bench + config arithmetic + journal.
- **No code rollback** (user rule, 10-17): a content/perf regression is LOCATED (to the
  operator/code path via dumps vs reference) and FIXED IN PLACE, then re-verified by
  `sm70_tests/correctness.py`. "Roll back to hide the bug" is not a fix; the valve
  rollback above means reverting an experimental valve, never source code.
- **Serve 重启的看门狗风险（2026-10-18 事故；内存归因已撤回）**：本机软件看门狗在心跳 stale>120s 时 `reboot -f`。
  12:28 那次重启的**触发仍未定论**——经 cgroup `memory.stat` 实测，稳态非可回收占用仅 **~61 GiB / 123 GiB**
  （anon ~25 + shmem ~30），`MemAvailable` ~61 GiB，`vmstat` si/so=0 ⇒ **不是 host RAM thrash**
  （我先前两次算术都错，更正见 `00_machine_state.md` / `03_history.md`）。规则：
  ① 重启前看 `MemAvailable`（`/root/run/ram_guard.sh`，<30 GiB 才告警，swap 残留不算）；② **两次重启之间冷却 ≥3 min**
  （理由是旧 worker/端口清理，**不是**防内存打满）；③ **每 session boot 次数要克制**（降低重启期 CPU/调度争抢饿死
  SCHED_FIFO 心跳线程的概率）；④ 重启后核对上一次 boot 是否异常中止。
  ⑤ 客户端足迹分级：`qsa_conc_tput`（线程 + 无 torch）**随便跑**；带 `transformers` 的（`baseline.py`、
  `srv_metrics_tps.py`，~2–4 GB）**必须串行**；`torchrun -8`（~16–24 GB）只在静默窗用。
  详见 `00_machine_state.md`。
- **Service bring-up: stream, never blind-wait** (user rule, 10-18). Starting
  `sm70-serve` behind a fixed-sleep poll loop is banned: a restart is DONE only when the
  journal shows a decisive marker, and a failure must surface immediately. Use
  `/tmp/opencode/serve_up.sh` — it streams `journalctl -u sm70-serve -f -n 0` and returns
  on the first hit: ready `Application startup complete.` (exit 0), or failure
  `Worker proc … died unexpectedly` / `Engine core initialization failed` / `CUDA error` /
  `Traceback` (exit 1 + error tail). `curl /health` is a cross-check, never the gate;
  `systemctl is-active` must be `active`. Every command chain that restarts the service
  must abort on this non-zero exit instead of running the next step blindly.
- **Diagnosing the captured (≤64) path** (10-18): `cudagraph_capture_sizes` max = 64, so
  ≤64-token prefill replays a breakable **FULL** cudagraph while >64 runs eager — that is
  the 64/65 boundary, NOT a kernel M threshold. In FULL mode `eager_break_during_capture`
  is bypassed, so **at replay no Python runs**: host-side probes in the model/QSA layer
  only ever observe the *capture* pass. To observe replay state, accumulate on device
  (`vllm/models/qwen4_exp/nvidia/sm70_nan_probe.py`) and dump from an eager forward.

## Records
- Durable facts/decisions/standards live in the `sm70/` docs (indexed by AGENTS.md).
  **Do not create new scattered .md.** Per-iteration working notes → a PLAN file only
  if it will be reloaded; otherwise it goes in `sm70/03_history.md`.

## Captured kernels must be batch-shape agnostic (P0#12 lesson, 2026-10-18)

- `qwen4_exp/common/ops/ple.py` `_ple_conv_kernel` carries **`MODE: tl.constexpr`**
  (decode vs prefill). A kernel like this is *specialized at capture time*: a
  breakable PIECEWISE graph captured with decode-shaped dummy input silently runs
  the decode branch when a prefill-shaped batch replays it ⇒ corrupt content.
  That was P0#12 (every ≤64-token prompt corrupted; the 64/65 boundary comes from
  `max_num_seqs`-derived staging/matching, NOT from any kernel M threshold).
- **Rule**: either make the mode a *runtime* argument (one graph valid for both
  shapes), or make the graph replayable only for the shape class it was captured
  for. The tree's own knob for the latter is `VLLM_SM70_GDN_GRAPH_DECODE`
  (defaults **off**, while `VLLM_SM70_GDN_GRAPH` defaults **on** — an inconsistent
  default pair). Fixed 2026-10-18 by gating in `cudagraph_utils.find_desc`:
  when GDN is graph-captured, prefill/mixed batches refuse PIECEWISE descriptors
  (decode matching untouched ⇒ decode perf unchanged).
- **Audit (2026-10-18)**: `MODE: tl.constexpr` decode/prefill specialization exists
  only in `qwen4_exp/common/ops/ple.py`; the other `_cudagraph_support` builders
  (`ALWAYS` / `UNIFORM_BATCH`) do not currently bake such a mode. **Re-run this
  grep whenever a new captured kernel is added** (a captured kernel that branches on
  decode-vs-prefill is the recurring failure mode).

### 离线 bench 的**时点判据不够**（2026-10-18 事故教训）
- `quiet_window_check` 已要求：服务日志/journal 近 60 s 无请求 + GPU 不忙 + 10 s drain ✓ **且它确实通过了** ✓
- 但它是**时点**判据：本次 bench 于 17:24:58 通过检查后开始，**17:25:48 外部客户端到达**（在 bench 进行中），
  17:25:50 worker-5 静默死亡、引擎自停、teardown SEGV ⇒ serve 崩溃（已恢复，二进制/配置零改动）✗
- **规则**：① 任何 **>60 s** 的离线 bench，**每 30–60 s 复核**服务 health 与近期请求；一旦有流量 ⇒ **立即中止**、
  该轮数据作废 ✓ ② 中止后**必须**检查 serve 是否仍 `health 200`，异常则按恢复流程处理并记入 `03` ✓

### 离线/JIT 实验必须隔离构建目录（2026-10-18 新规则）
- 本树有 **JIT 扩展**（如 `vllm_sm70_gdn_chunked_fwd`，源码 = `csrc/.../gdn_sm70_chunked.cu`）：实验改该 `.cu` 会
  **重建并写进共享的 torch 扩展缓存**；serve 在下次用到该扩展时会**加载那份缓存** ⇒ 实验代码等于进了生产 ✗✗
- **规则**：一切离线实验进程必须设 **`TORCH_EXTENSIONS_DIR=/tmp/opencode/ext_<实验名>`**（私有目录），
  绝不触碰生产缓存；实验结束还要确认 `.cu` 已回退到与基准**逐字节一致**（`diff` 为空）✓

### ⛔ 编辑"服务会 import 的代码"的硬规则（2026-10-18 事故 #2 根因）
- **事故**：对 `sm70_gdn.py` 一次**未加断言**的 `str.replace()` 只改到函数定义、没改到调用点
  ⇒ 模块自相矛盾 ⇒ 服务器下一个请求即 `NameError` ⇒ 8 worker 全崩 + teardown SEGV ✗
- **规则**：
  ① 任何对**运行时会被 import** 的 py 的改动，**每一处替换都必须断言**（`assert count==1`），
     绝不允许"部分生效"；② 改完**立即**跑 `py_compile` + **真实 import** + 校验新增符号存在/残留符号为零 ✓；
  ③ 改完**立即**验证服务（`health` + **一次真实请求**，且请求要走到被改的代码路径 ✓）；
  ④ 运行中的服务若已 import 旧模块，**必须重启**才算生效（否则"文件已改、内存未改"的窗口是个定时炸弹 ✗）

### ⛔ 实验阀不得在服务内触发 JIT 构建（2026-10-18）
- 本树的 GDN 内核是 **JIT**（`load(name=..., sources=[.cu])`）：若阀让服务去加载一个**尚未构建**的变体模块，
  就会在**服务进程内**启动 nvcc（分钟级 + 数 GB 内存）⇒ 服务卡死/被杀 ✗
- **规则**：变体模块**预先**在**服务自己的** `TORCH_EXTENSIONS_DIR` 里构建；或直接把**预构建 .so** 用
  `importlib` 加载（无构建路径），并在**离线**验证"阀开/关都不产生构建" ✓

## 政策 v2（2026-10-19，用户指令：解除全部限制、持续迭代）

**范围**：配置 / 并行布局（TP·DP·PP）/ 模型与量化 / 稀疏预算 / **profiler（nsys·ncu·torch profiler）** /
内核重写 / 多流与微批 / 投机解码（MTP）**全部放开**；旧的"禁改配置、禁改布局、禁碰模型"作废。

**判据**：`65K ruler (salt 9502)` + 内容门（`correctness.py` + `long_content_gate.py`）+ **0 新 Xid**；
**取消"单次 ≥3%"门槛**（小步快跑），但每次改动必须备份、过门、留回滚点，且不得把未验证的实验留在生产态。

**保留的安全/正确性纪律**（非性能限制）：
① 改前备份（脚本/源码/.so） ② 每次 boot 前 `ram_guard.sh` ③ **先证生效**（canary/日志/sha）再读数
④ **基线同期测**（记忆锚会漂 ±2%；已两次踩坑） ⑤ 补丁"先断言全部锚点、再一次性写回"（已两次踩坑）
⑥ 跨进程绝对值不可比（clocks 1290↔1530 MHz）⇒ 消融/相位同进程交错取 min ⑦ 实验室改动可回滚，已知 bug 原地修。

**新增可复用仪器**
- **`--enforce-eager` + `/tmp/sm70_opattr_mode`**：prefill 本来就 eager ⇒ 打开后 opattr **全 coverage**
  （实测总量 182.6 s / 16 = 11.4 s ≈ 墙 11.08 s ✓）。这是**唯一**能给出可信 prefill 五区分解的路径。
- **形状地板 bench**：按 checkpoint 里真实权重形状用 cuBLAS 量地板（例：HC down/up = 12.78 ms/GR 单卡
  ⇒ TP8 每 rank ~154 ms = 1.4% 墙）+ 与实测份额对比 ⇒ 直接判"是否离地板远"。
- **JIT 相位/消融**：`VLLM_SM70_GDN_PHASE/PROF`（K1/K2/K3）+ `__device__ g_*_abl` 门控（K1/K3 已用）。
- **QSA 武装通道**：`kv_cache_dtype` 的 `@N` 后缀 ⇒ C++ 解析成内核 `abl`（`ops/qsa.py::_abl_kv_dtype`，惰性默认关）。

**prefill 地图（同上 AGENTS §3 表）** 与 **decode 事实**：conc1 16.4 ms/步、conc8 27 ms/步（8 请求只慢 1.65×
⇒ 机器 ~80% 闲置 ⇒ **延迟/串行受限**，非算力/带宽）⇒ decode 的杠杆是 **MTP + 微批/多流 + DP 副本**，不是微调内核。

### 铁律（2026-10-19，用户定立）：**禁止用 /health 探测服务状态**
- 一律 **实时读 unit 日志**：`journalctl -u sm70-serve`。
- 成功判据 = 出现 `Application startup complete`；失败判据 = 出现 `Traceback` / 具体 `...Error:` 行 / unit 进入 `failed`。
- **失败时必须打印原始日志尾部**（真实异常行：`AttributeError` / `RuntimeError: Worker failed with error '...'` 等），
  不得只看 API 层的包装异常、也不得靠 `curl /health` 的 000 去猜。
- 统一入口：`/root/run/serve_wait.sh [timeout_s] [unit]`（exit 0 成功 / 1 失败并打印 RAW TAIL / 2 超时）。
- 本 session 教训：一次 `AttributeError`（我的探针在 `use_combine=False` 的 mixer 上取属性）被包装成
  "Engine core initialization failed"，靠 **原始 worker 行**才定位 ✓。

### 纪律 ⑩（2026-10-25 用户指令）：**脚本一律后台跑，定时短命令查**

- 长任务必须 `setsid nohup <script> ><log> 2>&1 &` 脱离会话启动。前台长跑时工具调用被中断会把整个进程组带走
  （实测两次：TP4 双实例被中断杀成 8 卡全 0 MiB，其中一次还误把上一代日志的 "A ready" 当成正在加载）。
- 等待用 goal_wait / 定时唤醒；**不要**在命令里 `sleep` 长等，也**不要**前台阻塞。
- 判进度只认 **GPU 显存 + 进程年龄**；日志行不能当证据（同一 log 文件跨 run 追加）。
- serve/daemon 顺序（`sm70-serve` 带 `Requires=sm70-daemon`）：先 stop serve 并等 cgroup 排空 → start daemon →
  等 READY（锚定 unit 启动时间戳；READY 行比 `ActiveEnterTimestamp` 早约 2 s，用 `@(EPOCH-60)`）→ `systemctl start serve`。
  不要在 serve 启动事务排队时 restart daemon（会取消 daemon 的启动任务，实测 7 分钟一次重载循环）。
- 旁证：双实例 TP4 的 `--gpu-memory-utilization 0.95` 没有余量（KV 已由 `--kv-cache-memory-bytes` 钉死，该值只当预检门），
  任何新增的初始化占用都会让第二臂直接 `ValueError: Free memory ... less than desired` ⇒ 已调成 0.93。

### 纪律 ⑨（2026-10-19 用户指令）：**失败实验不立刻恢复生产，只维护基线快照**
- 基线脚本快照：`/root/run/run_sm70_serve_ipc.sh`（各阶段备份 `/tmp/opencode/serve_pre_*.sh`，最近一次 =
  `/tmp/opencode/serve_pre_ep.sh` = 无 EP/无 HC_TP/无 spec/无 eager 的生产态）· daemon 脚本 `run_sm70_preload.sh`（8192、TS8）
- 基线性能锚（salt 9502，同期测）：~~prefill 65K hdr **5875–5923** · decode1 **60.9 tps** · conc8 **≈295 tps agg**~~
  **已作废**（旧模型/旧 wire 时代；同样作废的还有 `6055–6066 / 66.0–66.9 / 312–317`）。
- **当前锚（2026-10-24 复核，flat AR + FP4-Local-NVFP4；salt 9502/9503）**：prefill 65K hdr **8687**
  （8687/8705；hier 8684/8691）· decode1 65K **109.6 tps**（MTP k=1，中位 n=5，98.0–112.0；hier 100.9；
  无 MTP ≈80）· conc8 65K **404.7 agg**（hier 355.2）· canary **9/9** · Xid **0**。
  权威出处：`19_two_layer_ar_protocol.md` §8/§10。此后所有迭代一律与这组锚比。
- **新原点锚（2026-10-21 起，wire=fp16(3) 固定，禁止回滚；W2 融合后）**：prefill 65K hdr **6055–6066** ·
  prefill8k **6485–6508** · decode1 **66.0–66.9 tps** · conc8 **311.9–316.9 agg** ·
  ONLINE_CONTENT 6/6 · 长内容门 PASS · Xid 0。
  数据：原点臂 `/data/sm70_results/ar_wire_fp16/wirefp16/`、W2 `/data/sm70_results/ar_wire_fp16/w2_fused/`。
  与 fp32 相比：prefill 持平、decode1 6.0→66.0 起（spread 收窄到 1.4%）、**conc8 +1.6~1.8%（已反超 fp32）**。
  此后所有迭代读数一律与这组锚比（不再与 fp32 锚比）。
- 实验可以**连续叠加**：每次只读日志判成败、记录真实错误行，**不恢复、不重启回滚**；批次结束或需要生产窗口时再一次性恢复基线并过 canary。
- 这条把"每次失败 boot 的 5–10 min 恢复"从流程里去掉（本 session 已因反复恢复浪费了大量时间）。

### 纪律 ⑨ 修正（用户 2026-10-19）：**sm70-serve 是测试台，不是生产**
- 无人在线使用 ⇒ **停止"每次失败就恢复 + health 探测"的仪式**；只保留**基线快照**（脚本 + 性能锚，见上）。
- 试验流程 = 改 → boot → **读日志**判成败 → 测（ruler/canary）→ 记录 → 下一个；失败**不恢复**，只记真实错误行。
- ⚠️ **本条已被代码核验否掉（2026-10-19，六路只读审计）**：原写"服务端配置变更 ⇒ loader 自动回落磁盘 ⇒ 不必重建 daemon"**不成立**：
  ① **IPC 指纹不含调度器/缓存配置**（`weight_cache/protocol.py:283-325`：只有 tp/dtype/quant/quant_hash/ple_dir/ckpt/arch/pp/dp/moe_backend），
    而 **daemon 用 `--max-num-batched-tokens 8192 --max-num-seqs 8`、serve 用 16384/64** ⇒ 由这些常量定尺寸的缓冲（如 QSA `topk_indices_buffer` 是
    `register_buffer(persistent=False)`、会被 daemon 经 IPC 送出）**会不一致** ✗；
  ② **回落路径本身会崩**：`_fallback_load` 在同一进程二次构建 ⇒ 撞 `compilation_config.static_forward_context` 同名注册
    （全库 ~40 处）⇒ `ValueError: Duplicate layer name` ✗（补丁 `/tmp/opencode/fix_ipc_fallback.py` 已备未应用）。
  ⇒ **凡是改 `max_num_batched_tokens` / `max_num_seqs` / `block_size` 相关的量，必须把 daemon 一起改并 `restart`**，不要依赖回落。
  ⇒ 失败的**真实后果**见 03_history「chunk-16K 的 Xid 31：新机制」——`conv_batch_ptr` 这类按引擎 MNT 定尺寸的缓冲会在图捕获后 `resize_`，
    源码注释原文即"*Xid 31 on the first prefill replay*"。
- 恢复基线的场合仅两种：①需要一次干净基线对照 ②要交付/对外服务。

## 🚫 进程与服务操作铁律（2026-10-19 追加，均由本 session 事故换来）
1. **禁止 `pkill` / `killall`**（含"括号法"）✗ —— 本 session **两次**把自己的 shell SIGTERM 掉
   （第二次即使写成 `pkill -f "env_ab\.s[h]"`，pkill 仍把参数当正则、匹配到自身命令行 ✗）
   ⇒ **唯一允许的清理方式**：`ps -eo pid,cmd | grep <模式> | grep -v <自己的PID> | awk '{print $1}'` 取 PID
   ⇒ 再 `kill <PID>`（精确、可审计）✓
2. **Python 生产模块改动后，必须先做导入自检** ✗ —— 只做 `ast.parse` 不够（它只查语法、**不查名字解析** ✗）
   ⇒ `python -c "import vllm.distributed.device_communicators.<mod>"` 通过后，才允许 boot ✓
   （本 session 事故：往 `sm70_hier_ar.py` 插 `os.environ` 却无 `import os` ⇒ `NameError` ⇒ **每次 boot 都失败** ✗）
3. **结构性编辑（插入/包块）必须先编译再宣称** ✗ —— 用 `if (false) {` 占位曾破坏 if/else-if 括号 ⇒ 80 个编译错误
4. **同一时刻只允许一个 boot** ✗ —— 两个 harness 并发 `systemctl restart` 会互相踩踏（即使不是每次故障的真因）
5. **判据/比较逻辑必须先离线自测**（构造正/反例各一）✗ —— 否则"判据失效"会伪装成"实验失败"

## 📶 通信精度的选择纪律（2026-10-19 用户定稿）—— **带宽够 ⇒ 优先高精度；带宽不够 ⇒ 量化**
### 规则
1. **先判"瓶颈类型"，再决定精度** —— 不是"能省就省"，也不是"无损优先" ✗
   - **延迟受限**（小消息、链路利用率低）⇒ **不量化**，用高精度（fp32/fp16）走最低延迟路径 ✓
     · 实测：decode 5–160 KB ⇒ codec 的**逐元素/发射成本**远大于省下的字节 ✗ ⇒ 量化**纯亏** ✗
   - **带宽受限**（大消息、链路利用率高）⇒ **量化**，用最窄 wire 换字节 ✓
     · 实测：prefill ~40 MB/AR ⇒ 字节就是时间 ✓
2. **"够不够"的量化判据**（不要凭感觉 ✗）：
   - 看该次集合通信的**每 rank 有效吞吐 / 可达带宽**（bisection 归一或 egress 归一）
   - 若已接近上限 ⇒ 带宽受限 ⇒ 量化收益 = 字节比 ✓
   - 若远低于上限（如 <30%）⇒ 延迟/发射受限 ⇒ 量化只会更慢 ✗
   - **必须用服务内口径**（含计算抢占），standalone quiet-window 只作上界筛选 ✗
3. **精度阶梯与门槛**（按字节从少到多，先满足精度再谈字节）
   | wire | 字节/elem | 精度状态 | 门槛 |
   |---|---|---|---|
   | fp32 (0) | 4 B | 逐位可复现 ✓ | 无需 |
   | fp16 (3) | 2 B | **near-lossless** ✓ | 内容门 ✓ |
   | **e4m3 (1)** | 1.03 B | **已实测：正确性门 + 长门 双 PASS** ✓✓ | 内容门 + 长门 ✓ |
   | nvfp4 (2) | 0.56 B | **曾 CORRECT-FAIL + 长门垃圾** ✗（per-16 scale 固有不足 ✗）| **必须换 scale 方案（per-8/两级）并重过门** ✗ |
4. **结构前提（比选哪个 wire 更重要）** ✗✓
   - 量化只能作用在"**一个值量化一次**"的地方 ✓；**对"会被相加的部分和"量化会让误差在求和里叠加** ✗
     （这正是 nvfp4 wire 当年"塌"的机制 ✓；而权重/KV 的 nvfp4 是"一个值一次" ✓ 所以没事 ✓）
   - 量化必须**融合**（生产者 epilogue 直出 / 消费者直读）⇒ **零 staging、零额外内核** ✓
     否则 codec 的搬运成本会吃光字节收益 ✗（naive 版实测 0.83 ms > 整个 AR 0.76 ms ✗）
5. **分相位生效**（同一份代码两种策略）✓
   - decode（≤256 KB）⇒ wire=0 ✓ **一个字节都不改** ✓
   - prefill（>256 KB）⇒ 按上面的阶梯选 wire ✓
### 本机的具体数字（40 MB payload，20.97M 元素/rank；供快速判断）
| 方案 | 每 rank 流量 | vs NCCL | 结论 |
|---|---|---|---|
| NCCL ring fp16（现状 prefill）| 73.5 MB | 1.0× | — |
| 自研 hier, wire=0 fp32 | 168 MB ✗ | 2.3× | **必输** ✗（结构 2S + 4 B）|
| 自研 hier, wire=3 fp16 | 84 MB ✗ | 1.14× | **仍输** ✗ |
| **自研 hier, wire=1 e4m3** | **42 MB** ✓ | **0.57×** | **≈ +6% prefill** ✓（精度门已过 ✓）|
| 自研 hier, wire=2 nvfp4 | 21 MB ✓ | 0.29× | ≈ +12%，**但精度门未过** ✗ |
⇒ **结论：本机 prefill 必须量化才有意义（e4m3 起步）；decode 必须不量化** ✓

## 🔁 改动 → 需要重启什么（速查，2026-10-19 用户指出）
| 改动类型 | serve | daemon | 说明 |
|---|---|---|---|
| **AR 相关**：`AR_HIER` / `AR_WIRE` / `AR_WIRE_LARGE` / `AR_HIER_MAX_BYTES` / `/tmp/ar_hier_mode` / AR 内核 .so | **✓ 需要** | **✗ 不需要** | AR 的初始化和缓冲区全在 **serve worker** 内（`cuda_communicator` 里 `Sm70HierAR(...)`）✓；daemon 只做权重缓存 ✓ |
| QSA / GDN / MoE 的 env 阀 | ✓ 需要 | ✗ 不需要 | 同上（都在 serve 侧）|
| **调度参数**：MNT / block-size / max-num-seqs | ✓ | **✓**（纪律 k ✓）| 影响 ipc 缓存键与两侧调度一致性 ✗ |
| 模型/权重/量化配置 | ✓ | ✓ | daemon 也要按新分片重建 ✓ |
| 纯 Python 模块（如 `sm70_hier_ar.py`）| ✓ | ✗ | serve 进程重新导入 ✓ |
⇒ **AR 类实验的 boot 从 ~8 min 降到 ~2–3 min** ✓（只 `systemctl restart sm70-serve` + `serve_wait` ✓）
⇒ 注意：serve-only 重启在**权重配置未变**时走 ipc 缓存 ✓ 很快；若改了权重相关配置则会回落磁盘加载（更慢）✗

## ⏱️ 实验成本最小化（2026-10-19 用户要求：所有实验都按这个考虑）
**任何实验开始前，先回答"最小代价怎么做"，而不是默认全套 boot ✗**
### 四个杠杆（按节省幅度排序）
1. **最小重启范围**（见上面的速查表）
   - **不需要重启**：离线/standalone 微基准 · 服务内 profiler（start/stop ✓）· **per-launch 控制文件**（如 `/tmp/ar_hier_mode` ✓、`VLLM_SM70_MOE_TM_FILE` 的 R2-T1b 运行时翻转模式 ✓）
   - **只重启 serve**（~2–3 min）：AR / QSA / GDN / MoE 的 env 阀、serve 侧 Python 模块、serve 侧 .so ✓
   - **serve + daemon**（~6–8 min）：MNT / block-size / max-num-seqs / 权重与量化配置 ✗
2. **小尺寸筛选，大尺寸确认** ✓✓（本机最有效的一条）
   - AR 的**消息尺寸只由 batch 的 M 决定**（M×hidden），**与上下文长度无关** ✓
     ⇒ decode 类实验用 **len=8K**（AR 大小相同 ✓）而不是 65K（快 ~5× ✓）
     ⇒ prefill 类实验用 **len=16384**（chunk 仍被 MNT 限住 ⇒ AR 大小相同 ✓）而不是 65536 ✓
   - 只有**候选胜出者**才在 65K 上做正式判定（ruler + 内容门 ✓）
3. **一个 boot 测多个变体** ✓：能在同一 boot 内通过控制文件/请求参数切换的，就不要一 boot 一变体 ✗
4. **先离线后在线**：standalone/quiet-window 只作**筛选**（会低估/高估服务内代价 ✗），入围后再用服务内边际口径确认 ✓
### 时间预算参考（本机）
| 实验形态 | 单轮耗时 |
|---|---|
| 离线微基准（无服务） | 10 s – 2 min ✓ |
| serve-only boot + 小尺寸筛选 | **~3–4 min** ✓ |
| serve-only boot + 65K 正式判定 | ~6–8 min |
| serve+daemon boot + 65K 正式判定 | ~12–15 min ✗（仅改调度/权重时必要）|

## 🧨 事故：`Xid 154（Node Reboot Required）` —— 由"把 CUDA 客户端停到零"触发（2026-10-19 21:46）
### 时间线（原始证据）
```
21:46:58  systemctl stop sm70-serve   （为了跑离线带宽探针，想让 GPU 独占）✗
          内核 oops: remove_gpu / uvm_gpu_release_locked / uvm_va_space_destroy / uvm_pmm_gpu_deinit
                     kmem_cache_destroy uvm_gpu_chunk_5: Slab cache still has objects   ← nvidia_uvm 拆状态的 BUG ✗
          NVRM Xid 31  pid=modprobe  MMU Fault (ENGINE CE4 / HOST11)                     ← 随后的初始化踩到坏状态
          NVRM Xid 154 GPU recovery action -> 0x2 (Node Reboot Required)  × 8 张卡        ← 驱动判定不可恢复
          ⇒ serve 再也起不来（上下文创建失败；ipc 缓存失效后回落磁盘加载又撞 config 校验 ✗）
          ⇒ 只有【重启节点】能清掉 ✗（用户已重启 ✓，重启后 8/8 卡正常、Xid=0 ✓）
```
### 三个诱因（缺一不可，必须同时记住）
1. **persistence mode = Disabled** ✗（即便 `nvidia-persistenced` 显示 active ✗）⇒ 无客户端时 GPU 下电 ⇒ 走进那个有 bug 的 teardown ✗
2. **把 UVM 客户端停到 0** ✗（`stop` 而不是 `restart`）⇒ 正是 bug 的触发点
3. 这台机器本就在边缘：dmesg 里**已有 16 条 `name=VLLM::Worker` 的 Xid 31** ✗（不是我引入的 ✗）
### 新铁律（立即生效）
1. **禁止把 CUDA/UVM 客户端降到 0** ✗ —— 任何"完全停机"前，先起**占位进程**：
   `nohup python -c "import torch; torch.cuda.init(); input()" &` ✓（或只用 `systemctl restart` ✓，窗口极短 ✓）
2. **必须有 persistence mode** ✓：`nvidia-smi -pm 1` 并核对 `persistence_mode=Enabled` ✓（重启后重做 ✓，或用 unit/drop-in 固化 ✓）
3. **不允许为"独占 GPU"而停服务** ✗ —— 离线测量改为：
   **服务内探针**（`VLLM_SM70_AR_PROBE` 类 ✓）· 或**占位进程在场 + 小显存**的独立进程 ✓ · 或**完全离线的小尺寸**（几 MB，随便挤 ✓）
4. **重启后不要在前 30 秒判断"缺卡"** ✗：驱动是**分批**初始化 8 张卡（实测 5.9 s → 23.4 s ✓）
   ⇒ 判断"卡在不在"要等 `dmesg` 里 **8 条 `nvidia-drm ... Initialized`** 齐 ✓
5. **/tmp 是 tmpfs** ✗ ⇒ 重启即失 ✗ ⇒ **关键备份必须放 `/root/run/` 或 `/data/`** ✓（本次幸存的都在 `/root/run/*.bak_*` ✓）

## 📁 目录纪律（2026-10-19 用户明确）
| 位置 | 只允许放 | 禁止 |
|---|---|---|
| **`/root/run/`** | **服务运行配置**与其启停/等待脚本（`run_sm70_*.sh` · `serve_wait.sh` · `ram_guard.sh` · `warmup.sh`）· 配置回滚阀（`*.bak_*`）| **任何二进制/数据/日志/实验产物** ✗ |
| **`/tmp/opencode/`** | **一次性 scratch**（临时日志、随手文件）| **备份/阀门/需要跨重启存活的任何东西** ✗（实测：重启后被清空 ✗）|
| **`/data/sm70_results/`** | **持久产物**：原始数据 `<iter>/<exp>/` · 二进制/模型快照 `artifacts/` ✓ | — |
| **`sm70_tests/`** | **bench/测量脚本**（`bench_*.py` · `*_ab.sh` · 探针 `.cu`）| — |
| **`sm70/`** | **文档与结论**（00–09 + history）| — |
⇒ 本 session 违规记录：曾把 `.so` 备份写入 `/root/run/bak/` 与 `/tmp/opencode/` ✗ ⇒ 已迁至 `/data/sm70_results/artifacts/so/` ✓

### 2026-10-23 补充（再次违规 + 运维脚本被误删）
- **再次违规**：`serve_up.sh` 一直放在 `/tmp/opencode`（重启即失），今天的启动日志与一次性脚本也写在那里。
  用户再次明确：**禁止向 /tmp 写任何东西，除非真的是一次性使用**；脚本 / 阀门 / 日志 / 备份一律不得放 /tmp。
- **连带事故**：清理 `--kv-cache-dtype-skip-layers` 时，把 `/root/run/start_after_merge.sh`、
  `/root/run/wait_daemon_serve.sh`（正文含该 flag 字符串）连同 145 个历史脚本/备份一起删了
  ⇒ 冷启动只能临时重写脚本（重新引入检测/超时 bug）。
  **教训：清 flag 要 sed 掉那一行，不要删整个运维脚本；删除前先按功能分类。**
- **固定启停/检测脚本（唯一入口，禁止每次重写）**：

  | 脚本 | 用途 |
  |---|---|
  | `/root/run/sm70_up.sh` | 冷启动全栈：ram_guard → daemon（阻塞到 readiness gate）→ serve → serve_wait |
  | `/root/run/serve_up.sh` | 只重启 serve（daemon 已在跑）→ serve_wait |
  | `/root/run/serve_wait.sh` | 日志判定（`Application startup complete` vs 真错误行）；**禁止 /health** |
  | `/root/run/wait_sm70_ple_ready.sh` | daemon 的 ExecStartPost readiness gate |
  | `/root/run/ram_guard.sh` | boot 前内存闸门 |

  启动脚本持有 `/run/sm70_boot.lock`（一次只允许一个 boot；抢不到即 exit 3，不要重试）。
  日志固定写 `/root/run/sm70_up.log`。

## 🔒 "一次只允许一个 boot"已被代码强制（2026-10-19 再次违反后）
- 事故：我同时挂了**三个**会 `stop/restart` 服务的后台任务 ⇒ 互相踩踏 ⇒ 测出 `canary FAIL` 与
  `prefill 2046/3159`（半启动态的假数据 ✗）；**三项任务结束后重测 = 全绿** ✓（canary 9/9 · prefill 6059 · conc8 336.7）
- 措施：`sm70_tests/{env_ab,kernel_ab,kv_flag_ab,layout_ab}.sh` 全部加 **`flock -n /tmp/opencode/sm70_boot.lock`** ✓
  ⇒ 第二个任务会**立即退出（exit 3）**而不是踩踏 ✓
- 纪律：**任何会碰服务/GPU 的实验，启动前先确认没有别的任务在跑** ✓；被 `exit 3` 拒绝时**不要重试** ✗，先等前者结束 ✓

## ⛔ 铁律 R1：不接受任何回退策略（2026-10-20，用户定稿）

> "我不接受任何回退策略，这是铁律。"

**含义（无歧义）**
1. **不允许 fail-open**：任何实验路径初始化失败、参数不合法、库缺失、形状不整除 ⇒ **必须显式失败**
   （抛异常/拒绝启动），**不得**静默切到 NCCL/hier/其它路径。
2. **不允许 auto-rollback**：测量不达标的处置是**人工决策 + 明确记录**，脚本不得自动还原配置。
3. **不允许"兜底默认值"**：缺参数、缺环境变量、缺文件 ⇒ 报错；**不取默认值**继续跑。
4. **不允许 try/except 吞掉错误**：catch 只允许用于**清理+重抛**（re-raise），禁止 catch 后继续。
5. **不允许"保护性检查"替代正确性**：布局/索引必须**只有一份定义**、在主机侧被穷举验证
   （`sm70_tests/ar_ring_layout.hpp` + `test_layout.cpp`）；kernel 内不得出现布局算术。
6. **不允许在脏机器上运行**：Xid 154 / 未复位状态 ⇒ 拒绝运行（`gpu_guard.sh`）。

**为什么**：本 session 两次把驱动打死 + 多次把"未验证的结论"当结论，
共同点都是"出错后有一条软路径兜着"。软路径让错误**不痛**，于是错误被保留、被放大 ✗

**判定（自检问句）**：这段代码在出错的瞬间，是**立刻炸**，还是**偷偷换条路继续**？
后者一律不合格。

## ⛔ 铁律 R2：多卡 P2P 只能用生产方式（8 进程 + IPC handle）

**依据（用户 2026-10-20 直接质询）**："我之前适配也没搞死过 GPU" —— 成立 ✓

| | 生产 hier AR（安全，多年未出事故）| 我的离线 bench（今天三次 Xid 154）|
|---|---|---|
| 进程模型 | **8 进程，1 进程 1 context** | 1 进程 8 context |
| 跨卡内存 | **IPC handle**：`allocate_shared_buffer_and_handle` → `open_mem_handle` | `cudaMalloc` 指针直接跨设备解引用 |
| 合法性 | 每条腿都是**导入的、已注册的**映射 | 依赖 `cudaDeviceEnablePeerAccess` 让 UVA 恰好生效 |
| 失败形态 | 打不开就报错（进程内可捕获）| 映射不可见 ⇒ **HOST11 VIRT_READ ⇒ Xid 154** |

**规则**
1. 任何多卡 kernel 验证**必须**用 8 进程 + IPC handle（复用 `Sm70HierAR._alloc_ipc_buffers` 的模式）
2. **禁止**在单进程内对 `cudaMalloc` 出来的指针做跨设备解引用（`bench_ar_ring*.py`、`bench_wire_boundary.py` 当前是这种，标记为**不可再用**）
3. `gpu_guard.sh`、`wire_preflight`、主机侧证明（layout/codec/sim）继续保留 —— 它们是补充，不是替代


## 🧨 实验性常量绝不能留在 HEAD（2026-10-24 事故）

**经过**：GDN 的 `constexpr int C` 曾被临时改成 32 做实验（该值会让引擎报 CUDA 700 崩掉），
后来"回滚"只改了工作区、**HEAD 里仍是 32**。因为 GDN 是**运行时 JIT** 编译的，
下一次重启就把 32 编进去 ⇒ 服务起不来（canary 7 项 connection refused），
而现象看起来像新故障，实际是旧实验的残留。

**规则**：
1. **实验性常量/开关（C、tile、宏、env 默认值）改完必须立刻回滚并 commit**，
   不允许"先放着，等会儿一起回滚"；
2. **每次 commit 前 `git diff --stat` 核对**只包含预期文件（这次就是漏了这一步，
   把一个 `sed` 的临时值一起提交了）；
3. **恢复服务失败时，先查 `git status` 与关键常量，再看服务日志** ——
   服务日志只会告诉你"CUDA 700"，不会告诉你源码里有个残留的 32；
4. 改动 JIT 源码（`csrc/gdn/*.cu`、QSA 等）后**必须清 JIT 缓存**再重启，
   否则测/跑的是旧模块。

## 🔁 切换 TP 规格必须清 JIT 缓存（2026-10-24 实测）

`torch.utils.cpp_extension.load` 按**模块名**缓存（`vllm_sm70_gdn_chunked_fwd` 等），
TP4 与 TP8 共用同一个名字。跑过 TP4 之后直接起 TP8，会复用 TP4 期间编译的模块，
表现为 **canary 非确定性失败**：

```
determinism: len=322/318
prefix_cache: same_text=False
```

**处置**：切换 TP 规格前后都必须

```bash
rm -rf /root/.cache/torch_extensions/py312_cu128/*
```

再重启服务。实测：清缓存 → 重启 → **canary 9/9** 恢复。

同理，改 `csrc/gdn/*.cu`（JIT 源码）后也要清，否则测的是旧模块。

## 🔨 构建入口（2026-10-24 补写：之前文档没写，我踩了两次）

**改动 `csrc/` 后的唯一正确入口是仓库根目录的 `./build_sm70.sh`，不是 `setup.py` / `pip install`。**

```bash
cd /data/src/vllm-v2 && JOBS=32 ./build_sm70.sh          # 默认只编 _C_stable_libtorch + _moe_C_stable_libtorch
```

它已经处理好这台机器的坑：
- **钉住 `/usr/local/cuda/bin/nvcc`**（apt 的 CUDA 12.0 nvcc 解析不了 glibc 2.39 的 stdlib.h ⇒ configure 直接失败）；
- `BUILD_DIR=build-sm70`（**不是** `build/`），ninja + ccache 增量；
- `TORCH_CUDA_ARCH_LIST=7.0`；
- 依赖按钉定版本从本地 `.deps` 取（离线），只在版本 bump 时才联网。

反面教材（我犯的两次）：
1. `python3 setup.py build_ext --inplace` → 用系统 python，`ModuleNotFoundError: torch`；
2. `/data/vllm-dev/bin/python setup.py build_ext --inplace` → 没有 `build/` 目录 ⇒ 从零 cmake 配置，
   且它选的 nvcc 不对 ⇒ configure 失败。
两次都白跑，**任何 csrc 改动前先看根目录有没有 `build_sm70.sh`**。

## ⛔ 命令黑名单（2026-10-24 用户裁定）

**禁止使用 `pkill`**（已用 `/usr/local/bin/pkill` 垫片强制拦截，除非 `ALLOW_PKILL=1`）。
原因：`pkill -f <pattern>` 按**完整命令行正则**匹配，会杀掉命令行里恰好含该字符串的无辜进程：

- 2026-10-24：`pkill -9 -f "[v]llm.entrypoints.cli.main"` 同时命中 **serve** 和 **preload daemon**
  （daemon 的命令行也是 `python -m vllm.entrypoints.cli.main preload ...`），把 daemon 打死，
  丢了 IPC 缓存（重建 ~10 分钟）；
- 更早一次：`pkill -9 -f "peer_rt.py"` 命中了执行它的 shell 自己，把工具调用自杀。

**替代做法（必须）**：`pgrep -af '<pattern>'` 看清 PID → `kill -9 <pid>` 逐个杀；
或 `systemctl stop <unit>`。**永远不要用模式串批量杀进程。**

## 🧪 崩溃不等于实验结束（2026-10-24 用户裁定）

**一次实验跑崩（引擎死、超时、结果错），不算"这个方向不行"，只是"这个变体还没跑通"。**
只有同时满足下面三条才允许停：
1. 失败的**根因已定位**（不是"不知道为啥挂了"）；
2. 根因属于**不可修复**（硬件/语义限制，或修复成本远超收益）；
3. 已尝试**至少两种不同的修法**并各自有实测结论。

在满足这三条之前，正确的动作是：**抓失败模式 → 提一个可验证的假设 → 改一处 → 再测**。

反面案例（2026-10-24 GDN `C=32`）：引擎崩了、日志只显示 `shm_broadcast ... RuntimeError: cancelled`，
我当场就写"实测否决"。正确做法是继续查：是内核挂起（客户端超时→cancel）还是结果错、
`chunk_offsets`/`h_ws` 的全局 chunk id 在 C 变化后是否仍自洽、JIT 是否真的重编译了。

## 🔍 深入探索纪律（2026-10-24 用户裁定，覆盖"一次没赚到就放弃"）

**用户原话**：「任何一个优化策略，都放弃得太快，没有深入探索优化，可能会导致大量可优化路径！」

我的历史错误：AR 只试了 S（切片数）一个维度就宣布"并行度到顶"；HC up_gate 只试了 CTA 形状
（row4）一个维度就宣布"无解"；glue 只看了一眼站点就下"只值 0.02 ms"的结论。

**规则**：
1. **一个方向只有在"探索矩阵"被穷尽后才能关闭**，矩阵至少覆盖 5 个维度：
   ① 算法/分解（split-K、二分、分块顺序）② 并行度（CTA 数、线程数、warps、切片）③ 数据布局与对齐
   （SOA/AOS、向量宽度、bank/L2 友好）④ 缓存/预取/eviction 策略 ⑤ 发射方式（融合、合并发射、图）。
2. 每维度 ≥2 个变体，**关闭前至少 8 个独立实测点**，并且必须测到该维度的**最优变体**，
   不能只测默认配置就下结论。
3. 关闭一个方向必须提交**"试过的全部变体 + 每个的实测时间"**表格，禁止一句"无收益"了事。
4. 每轮探索都要有 CPU 预检（算术/边界/bit-exact），再在**停服窗口 bench** 或**服务内 profiler** 测，
   单变量、同 boot 配对；变体之间不重启服务。
5. 回滚仍只在"零/负收益且矩阵已穷尽"时发生；中性结果按复杂度一次性决定。

## ⛔ 探针安全（2026-10-24 事故二：在线服务 + 8 进程 peer access ⇒ Xid 154，全节点需重启）

- **serve 在线时，禁止任何多进程 peer-access / CUDA-graph 探针**（IPC 区域 + `cudaDeviceEnablePeerAccess`
  + graph capture 的组合会把驱动打到 Xid 154）。要测就先把 `sm70-serve` 停掉。
- 单进程、无 peer access 的离线测量（算术、nsys 解析、单卡 kernel bench）不受此限。
- Xid 154 的症状：`nvidia-smi` 正常但**新 CUDA context 失败**；此时 serve 重启会在
  `VllmConfig` 校验处报 `modelopt_mixed ... Current capability: 70`（能力探测异常 → SM70 例外失效）。
  处理：立即 reboot，不要在坏状态下反复重启服务。

## ⛔ 探针安全（2026-10-24 事故：SYS/PCIe 对把两张卡钉在 100%）

- **禁止压测 SYS/PCIe 对**（用户明确指令 2026-10-24）：`nvidia-smi topo -m` 里标 **SYS** 的对
  （如 0↔4/5/6）不走 NVLink，peer access 不可用/不可靠；探针一旦依赖 flag 握手就会**无限自旋**，
  本次把 GPU0/GPU4 钉在 100%，直到 SIGKILL 才恢复。
- **更正（实测 2026-10-24）**：`cudaDeviceEnablePeerAccess` 在 **SYS/PCIe 对上同样返回 0**（P2P 可建立），
  所以**返回码检查不足以保护手写 flag 协议**——跨 PCIe 的 `st.release.sys`/`ld.acquire.sys` 可见性不可靠，
  对端会永远轮询不到。必须**按拓扑拒绝 SYS 对**（`peer_rt.py` 已加 NV_NEIGH 白名单）。
- 生产 AR 之所以能用满 8 卡：8 卡分成两个**全 NVLink 的 4-clique**（{0,1,2,3} 与 {4,5,6,7} 内部两两 NVLink），
  跨 clique 走 4 条 **mirror 链接**（0↔7 / 1↔6 / 2↔5 / 3↔4，均 NVLink）——**SYS 链路从不承载 AR 流量**。
- 所有 GPU 探针必须：① 按拓扑白名单只打 NVLink 邻居对；② 再查 `cudaDeviceEnablePeerAccess` 返回码；
  ② 自旋等待**必须有上限**（本次已给 `peer_rt.cu` 加 20M 次上限）；③ 只测 NVLink 邻居对
  （实测拓扑：0:[1,2,3,7] 1:[0,3,2,6] 2:[0,5,1,3] 3:[1,4,0,2] 4:[3,6,5,7] 5:[2,7,4,6]
  6:[4,7,1,5] 7:[5,6,0,4]）。
- 探针进程异常退出后，务必 `nvidia-smi` 确认 8 卡 utilization 归零、`/v1/models` 与一次真实
  `/v1/completions` 正常、`correctness.py` 9/9 后再继续测。

## 🧠 Profiler 协议（2026-10-24 用户裁定：不能"采一次就重启服务"）

**必须用有界会话**：`/start_profile` 带 `{"max_iterations": N}`（默认 100 步），
profiler 自己停；**不要**用"跑一段再 stop"（客户端若还在生成长请求，stop 会被拖住，
200s 的会话把每个 worker 的匿名内存撑到 ~10 GB，只能重启服务回收）。

- 启停一律走 `/root/run/prof_guard.sh`（已改为 bounded：`ITERS` 环境变量或第 3 参数）。
- stop 之后 worker 会调 `malloc_trim(0)` 把释放的堆还给 OS（`VLLM_SM70_PROF_TRIM=0` 可关）。
- 采集前 available ≥ 20 GB；采集窗口内客户端输出 ≤ 1024 tokens。
- **activities 固定为 `["CUDA"]`**（serve 的 `--profiler-config`）：我们只用 `dump_cuda_time_total`
  的 key-averages，CPU 事件是纯粹的缓冲浪费（实测 100 步会话：CPU+CUDA 2.2 GB/worker → CUDA-only 1.2 GB）。
- 默认 **ITERS=20** 步：key-averages 足够稳定（20×48=960 次 MoE），而实测一次 20 步会话后
  worker RssAnon **不升反降**（上一次的残留被本次 stop 的 trim 收回），available 56 G→62 G
  ⇒ **采集不再需要重启服务**。
- 验收读数用 per-kernel key-averages（`*_rank*.profiler_out.txt`）+ `sm70_tests/prof_rank_map.py`。
- 采集时机：先确认请求进入稳态（65K prefill 约 8 s），再起 profiler；否则窗口会采到 prefill。

## ⛔ 达标纪律（2026-10-23 用户裁定）

**未达标 ≠ 回滚。** 目标是"做到达标"，不是"没做到就退回去"：

- 单轮实验未达验收线 ⇒ **继续定位并优化到达标**（换实现、换参数、换路径都行）。
- **回滚只用于三种情况**：① 服务不可用（起不来/崩溃）；② 正确性失败；③ 把未验证的实验留在生产态。
- 不得用"这一轮没提升/没达到门槛"作为关闭方向的理由；关闭方向只能来自
  **实测天花板**或**有据的结构性阻塞**（并附算式/证据）。
- 每次迭代仍要：备份 → 单变量 → 同协议实测 → 留回滚点 → commit；失败只记日志真行，立刻换下一个候选。

## ⛔ 否决纪律（2026-10-22 用户裁定，最高优先级）

**只有两种情况允许否决一个优化方案**：
1. **硬件限制**——有实测证据（带宽/算力/延迟/占用率）证明已达该硬件天花板；
2. **软件与算法绝对无法解决**——有据的结构性阻塞（例如语义上必须的依赖、不可绕过的正确性约束），
   且已列出候选解法并逐一说明为何不可行。

**除此之外，一律不许否决**：
- "A/B 结果更慢"**不是**否决理由——那只说明**当前实现**慢，要继续查是实现问题（并行度/遍数/访存/同步）
  还是接口接错，修到它该有的水平再比。
- 不许用"已立项/留作独立项"当否决的挡箭牌：立项只用于**排序**，不用于**关闭**。
- 每次想写"否决/放弃/记为已达极限"之前，必须先给出：①该实现的算术/带宽/延迟下界；②实测/下界比值；
  ③针对差距的具体改法（≥1 条）。缺一条就不许写否决。

**本纪律的由来**：`qsa_lexicographic_topk` 曾因 A/B 慢 7.5% 被我判"否决"；读码后发现是
**decode 路径只用了 1 个 block（80 SM 用 1 个）+ 固定 4 遍扫描** ⇒ 是实现问题，应修不该否决。

## 运维纪律补充（2026-10-22，因一次事故）

**禁止在服务运行时替换 `.so`**：本轮把编译出的 `_C_stable_libtorch.abi3.so` 覆盖到 `vllm/` 下时，
正在运行的 serve 引擎随即死亡（`sm70-serve: failed`，`/v1/models` 仍 200，只有真实生成请求才暴露）。
⇒ 任何 `.so` 替换必须：①先 `systemctl stop sm70-serve`；②替换；③`start` + `serve_wait`；
④**用真实 `/v1/completions` 验证**（不是 `/v1/models`）。回滚同样需要 stop→替换→start。
## 代码树纪律（2026-10-22 用户裁定，最高优先级）

**禁止开 git worktree。所有修改与上游合并一律原地进行，唯一在服代码树 = `/data/src/vllm-v2`。**

背景：10-22 的上游合并做在工作树 `/data/src/vllm-sync`（分支 `sync-main-2026-10-22`），systemd 的 serve/daemon drop-in 指向该工作树
（PYTHONPATH），而 base unit 与测试脚本仍指 vllm-v2 ⇒ 当天连环事故：daemon 有两份脚本（base `run_sm70_preload.sh` /
drop-in `run_sm70_preload_sync.sh`），只改了前者导致权重缓存 key 永不匹配；排障时在两棵树之间来回绕。

执行要求：
1. 收敛：把 `sync-main-2026-10-22` 合回 `sm70-v2`，drop-in 的 PYTHONPATH 指回 `/data/src/vllm-v2`，重启 daemon→serve，删除 `/data/src/vllm-sync`。
2. 今后合并上游/大改：仍在 vllm-v2 原地做；回滚点用 commit/branch + 备份脚本（/tmp/opencode、`/data/sm70_results/`），**不用 worktree**。
3. 文档与脚本不得再引用"两棵树"；一个 systemd unit 只允许**一个** ExecStart 脚本来源（base 与 drop-in 不得各指一份）。

## 🚫 禁止引用预编译/临时 `.so`（2026-10-24 事故，硬规则）

**唯一允许的内核来源**：本仓库 `csrc/` 经 `./build_sm70.sh` 构建出的
`vllm/_C_stable_libtorch.abi3.so`，通过 `torch.ops._C.*` 调用。

**明令禁止**（不是"不推荐"）：
- `importlib.machinery.ExtensionFileLoader(...).load_module()` 加载任何 `.so`；
- 引用 `/tmp/**/*.so`、`.superpowers/sdd/**/*.so`、`/root/run/**/build/**/*.so`；
- 任何"临时 build 更快/先跑起来再说"的路径。

**事故经过**：`flashv100` / `flashv100_tp8` 两个 `.so` 的源码在
`/tmp/opencode/flashv100_src{,_tp8}`（**已删除**），而 `_FLASHV100_DEFAULT_SO`
指向的 `/root/run/sm70/flashv100_build/...` **根本不存在** ⇒ 加载静默返回 `None`
⇒ page4 路线静默关闭 ⇒ 连续 6 轮测量被误读为"gate 条件/stride/补齐"问题。
**处置**：两个 `.so` 已于 2026-10-24 永久删除；两个 loader 改为只返回树内模块。

**强制预检（秒级，每次改内核路径/每次测量前）**：
```bash
/data/vllm-dev/bin/python -c "import torch, vllm._custom_ops as c; \
  assert hasattr(torch.ops._C, 'grouped_sparse_page4_plan_fwd'), 'tree ops missing'; \
  print('page4 abi', torch.ops._C.grouped_sparse_page4_abi_version())"
```
必须打印 `page4 abi 2`。失败 ⇒ **停止测量**，先修构建。
（注意：裸 `import vllm` 不会加载扩展，必须走 `vllm._custom_ops` 这条引擎导入路径。）

**代码层强制**：`_qsa_page4_local_ops()` 缺算子时打 **ERROR**
（"QSA page4 DISABLED…do not trust page4 numbers"）。
日志出现该行即视为路线关闭，**任何吞吐数字都作废**。

**几何约束（必须记住）**：树内 build 是 **4-head**（`kGroupedVerifyHeads=4`）；
TP4 是 6 head ⇒ 6-head 路线用**两次 3-head pad→4 调用共享同一份 planner**
实现，不得再引入第二个 head 几何的构建产物。

## ⛔ 不允许原地改写正在运行的脚本（2026-10-24 事故）

`bash` 逐行读取脚本；用 `python`/`write` 原地重写**正在执行**的脚本，
会让运行中的 shell 读到半写内容（本次导致 `tp4_prof.sh` 的 restore 段
变成 `null | grep ...`，TP8 没有被拉起，serve 直接 down）。
**规则**：改脚本前先确认没有同名进程在跑；或写新文件再 `mv`（原子替换）。

## ⏱️ TP4 测量"一趟出两件事"，不自动恢复 TP8（2026-10-24）

`/root/run/tp4_prof.sh` 现在的语义：
1. 停 serve/daemon → 起 TP4 serve（nohup，日志 `tp4_prof_serve.log`）；
2. profiler 跑 1 轮 65K prefill + per-kernel 表（吞吐）；
3. **同一 boot 内跑 `correctness.py`（TP4 canary，9 项）** —— 这是正确性门，
   因为 6-head page4 路线此前从未执行过；
4. 停 TP4，**测试台保持空闲**。

**不再自动恢复 TP8**：按 2026-10-19 政策，只有 ①需要干净基线对照 ②对外/交人，
才起 TP8。每次实验后花 10 分钟恢复 TP8 是纯浪费。

**脚本改动纪律**：写临时文件 → `bash -n` → `mv` 原子替换 → **`chmod +x`**
（本次 `mv` 丢了可执行位，导致一次启动失败）。

## 🧯 等待 daemon 就绪不能用 `--since "30 min ago"` 判定（2026-10-24）

**事故**：gate 脚本用
`journalctl -u sm70-daemon --since "30 min ago" | grep -q "Weight cache daemon READY"`
判断就绪。窗口里**上一轮 daemon** 留下的 READY 行让循环提前退出 ⇒ serve 在
daemon 仍在加载时启动 ⇒ worker 报 `ValueError: Duplicate layer name: language_model...`
⇒ `serve_wait FAILED`，白跑一次 boot。

**规则**：等待条件必须锚定**本次启动时间**：
```bash
DAEMON_START=$(date -Is)
systemctl start sm70-daemon
... journalctl -u sm70-daemon --since "$DAEMON_START" | grep -q "Weight cache daemon READY"
```
并且要先 `systemctl is-active --quiet sm70-daemon` 再谈 READY。

**另一个同类坑**：`pgrep -f "vllm.entrypoints.cli.main serve"` 抓不到 systemd 起的进程
（ExecStart 是脚本）。取 PID 用 `systemctl show -p MainPID --value sm70-serve`。

## 📦 备份与阀门不得放 `/tmp`（再次强调，2026-10-24 违规）

本轮把 serve 脚本备份写进了 `/tmp/opencode/`（tmpfs，重启即失），与既有规则冲突。
**已迁至** `/data/sm70_results/artifacts/serve_pre_budget16.sh`。
一次性日志可以放 `/tmp`；**脚本/阀门/备份一律放 `/root/run/` 或 `/data/`**。

## 🔁 重启服务必须"先停透、再启动"（2026-10-24 两次事故）

`systemctl restart sm70-serve` 在旧进程组尚未完全退出时返回，紧接着新引擎启动 ⇒
两代引擎在同一节点上并存 ⇒ worker 报
`ValueError: Duplicate layer name: language_model...`（本次连续两次踩中）。

**规则**：统一用 `/root/run/serve_restart_safe.sh`：
`systemctl stop` → 等 unit 变 inactive → 等 `pgrep -f vllm.entrypoints` 归零 → sleep 10 → `systemctl start`。
禁止用 `restart`，也禁止并发多个会停/起服务的任务（标准 §"三个后台任务互相踩踏"）。

## 🧩 "Duplicate layer name" 的真因：IPC 回退后二次建模型（2026-10-24，三次踩中）

表象：worker 启动即 `ValueError: Duplicate layer name: language_model.model.layers.0.mlp.experts`。
**真因在上一行**：

```
Weight cache IPC loading failed; falling back to disk loading      ← 先出现
ValueError: Duplicate layer name: ...                              ← 只是后果
```

IPC 加载失败 ⇒ loader **在同一进程里回退到磁盘加载并再次构建模型** ⇒
`register_layer_for_moe_forward_op` 发现层名已注册 ⇒ 抛错。
而 IPC 失败通常是因为**上一轮 serve 被非干净地杀掉**（`Failed to kill control group`），
把 weight-cache 共享通道留脏。

**规则**：
1. 每次测速 boot 走 `/root/run/serve_cycle_clean.sh`：停 serve → 等 inactive 且
   `vllm.entrypoints.cli.main serve` 归零 → 停 daemon → 起 daemon → **锚定时间戳**等 READY →
   `ram_guard` → 起 serve；
2. 看到 `Duplicate layer name` **不要**去查 MoE 注册逻辑，**先往上找**
   `Weight cache IPC loading failed`；
3. `pgrep -f "vllm.entrypoints"` 会同时命中 daemon（`...cli.main preload`）
   ⇒ 必须用 `vllm.entrypoints.cli.main serve` 精确匹配。

## 🚦 规则不是散文，闸门才是（2026-10-24，回答"为什么总是不遵守"）

**症结**：同类操作错误反复出现（自匹配 `pgrep`、多余重启 daemon、旧 journal 窗口、备份写 /tmp），
因为规则只写在文档里，**犯错的那一刻没有任何东西会拦住**。写第 N 条散文没用。

**机制**：所有测量/启动脚本一律通过
```bash
/root/run/run_checked.sh [--preflight-only] <script> [args...]
```
执行。它会扫描脚本并**拒绝执行**以下已知坑：
1. `pgrep -f "<字面量>"` 且该字面量在脚本别处也出现 ⇒ 自匹配风险（要求 `x_[y]` 形式）；
2. `systemctl restart sm70-serve` ⇒ 必须用 `serve_restart_safe.sh`；
3. 用 `--since "N min ago"` 判 daemon READY ⇒ 必须锚定启动时间戳；
4. 备份 `cp` 到 `/tmp` ⇒ 必须 `/data/sm70_results/artifacts`；
5. 仅 serve 侧改动却调用 `serve_cycle_clean.sh`（含 daemon 重启）⇒ 提示。

**验证过**：违规样例被 REFUSED（exit 2）；`--preflight-only` 只检查不执行。

**两条新规则（本轮踩到）**：
- **spec/MTP 配置只在 serve 侧** ⇒ 只重启 serve；daemon 的 IPC 指纹不含 `num_speculative_tokens`，
  重启 daemon 只是白花 6–8 分钟载权重；
- **`pgrep -f "脚本名"` 会匹配到执行该命令的 shell 自身**（命令行含该串）⇒ 用 `pgrep -af 'k3_[c]alib'`
  或 `systemctl show -p MainPID --value`。

## 🧾 2026-11-01 错误教训登记（用户逐条指出，全部为已发生的真实反转）

> 背景：这一轮做 TP2 适配 + 长上下文 attention 分析，连续出现口径错误。用户要求"记录所有错误教训"。
> 每一条都写成**可执行的闸门**，不是散文。

### E1 用"重启服务"做两臂消融（最贵的一次）
- **事实**：`VLLM_SM70_ATTN_SKIP` 在 `try_paged_prefill_twopass()` 里**每次调用现读** `os.environ`，
  一个进程内交替 0/1 即可完成 A/B；我却为两臂各起一次完整 vLLM（每次 ~3 min 加载 + 71 s×N）。
- **代价**：双倍时间、跨进程漂移、还占着服务不可用。
- **闸门**：改 env 做 A/B 前，先 `grep` 这个 env 的读取点——
  **import-time（模块级常量）才需要重启；call-time（函数内 `os.environ.get`）一律同进程交错 min-of-N。**

### E2 把"重复 prompt"当冷启动测（产生 421 ms 的假数据）
- **事实**：把随机盐放在 prompt **尾部**，prefix cache 以 16 token 的 block 为单位，
  只有最后一个 block 失效 ⇒ 只重算几个 token，220K "冷启动"量出 **421 ms**（≈ 520K tok/s，物理不可能）。
- **闸门**：冷启动请求必须让**每个 block 都不同**——**每句**加随机前缀（`one_len_text.py` 的写法），
  并且**断言响应里 `cached_tokens == 0`**。

### E3 把 init 后的第一次请求当有效测量
- **事实**：服务刚起来时分配器/页表/时钟未稳，第一次请求偏慢或偏快。
- **闸门**：**先热身一次、丢弃**，之后取 2–3 次的**中位数**。用户原话："冷启动第一次请求才是错误的测量。"

### E4 结果没有过"物理合理性"闸
- **事实**：日志里明明出现 220K prefill = 421 ms，我仍然让 `min()` 把它选成最优。
- **闸门**：任何 TTFT/tok/s 先算**上限校验**（65K 目前 ≈11 s ⇒ 220K 不可能 < 8 s；
  tok/s 不得超过硬件可解释范围），越界值直接判为"测错"并丢弃，绝不进 min/median。

### E5 把测试脚手架的 bug 当成被测对象的 bug
- **事实**：早前验证 `fa_pf` 在 H=1 时"输出全零"，实际是我把 `A["out"]` 传给了 kernel、
  却去比较另一个未写入的 `o_ref` 张量，于是误判"TP2 有正确性空洞"。
- **闸门**：新测试先用**已知正确的对照形状**（如 TP1 的 (H,HQ)=(2,16)）跑同一段代码，
  对照必须通过；对照不过 = 脚手架错，不是被测错。

### E6 用 `utilization.gpu=100%` 论证"没空转"
- **事实**：等内存的 kernel 也计 100%。正确判据是 **power.draw + clocks.sm**（实测 280 W/300 W、1517–1530 MHz）。
- **闸门**：性能归因必须带功耗/时钟证据；util 只能作为辅助。

### E7 Python 层插桩/消融对 cudagraph 盲
- **事实**：`apply_scaled_mm` 在 cudagraph 重放时根本不执行 ⇒ 日志只记到 warmup；
  "跳过 dense GEMM"两臂只差 6 ms（阀从未生效）。
- **闸门**：**组件级归因用 eager 消融**（两臂都 `cudagraph_mode=NONE`）；per-kernel 用 CUDA-only profiler。

### E8 profiler 按内核名分桶会跨组件
- **事实**：完整 CUDA-only profile 把 cutlass 桶记了 3.744 s，消融实测 dense 只有 2.08 s；
  同时把 attention 的 QK/PV 归到 cuBLAS 桶，低估 attention。
- **闸门**：**份额以消融为准**，profile 只用于内核级参考。

### E9 不同 m / 不同输入 / 不含附属开销就下结论
- **事实**：曾把 +21% 回归误判为 −11% 胜利。
- **闸门**：两条路径比较必须**同 m、同输入、含全部附属开销**，否则结论作废。

### E10 TP 下按"形状"判断张量取向
- **事实**：TP2 的 `RowParallelLinear` 分片是**方阵**，`(out,in)` 与 `(in,out)` 两个分支同时成立，
  第一支命中 ⇒ 权重多转置一次 ⇒ 30 层 GDN + 10 层 attention 全错（整体乱码）。
- **闸门**：取向只能来自**显式元数据或单一约定**；方阵必须 fail-closed（见 §13.69）。

### E11 用"单点测量"当曲线（本次 250K 只测了 1 次冷启动）
- **事实**：250K 的 82.11 s 是单次冷启动；虽与模型自洽，但没有中位数支撑。
- **闸门**：长度曲线每点 **≥2 次冷启动取中位**；单点只能标注为"初值"。

### E12 多进程（TP>1）下改父进程 env 传不进 worker
- **事实**：TP2 的注意力阀 `VLLM_SM70_ATTN_SKIP` 用 `os.environ[...]=` 在父进程逐次翻转，
  两臂结果完全相同（71.19 vs 71.31 s）——模型在 spawn 出来的 worker 里，**继承的是 spawn 时刻的 env**。
- **正确做法**：用 `llm.apply_model(fn)` 在 worker **内部**改 `os.environ`（见
  `/tmp/opencode/ablate_attn_worker.py`），否则只能"启动前定死"。
- **同一个坑的第二层**：传给 `apply_model` 的函数必须定义在**可导入模块**里；
  定义在 `__main__` 会报 `Can't pickle <function ...>: it's not the same object as __main__.fn`
  （worker 重新导入 `__main__` 得到的是另一个对象）。见 `/tmp/opencode/valves.py`。
- **闸门**：任何"逐次翻 env"的 A/B，先确认 `tensor_parallel_size == 1`；
  TP>1 一律走 `apply_model`，并**先用一个已知有大差值的阀做阳性对照**（差值≈0 ⇒ 阀没生效）。

### E13 并发跑两个吃卡的 job → 初始化失败被误判成"代码坏"
- **事实**：旧实验的 `skip` 臂仍占着 2×30 GB，新任务初始化报 `Engine core initialization failed`；
  表象像代码问题，实为资源冲突。
- **闸门**：起任何离线任务前先断言 `nvidia-smi` 每卡 0–4 MiB，并确认没有同名进程残留；
  按 PID 终止父 shell 与子进程，否则脚本会自动再起一臂。

### 测量纪律补充（2026-11-04，已被骗两次）

**任何对 `_CUDA`/`_XQA_CUDA` 等内联 CUDA 源的改动（包括只改注释）都会让
`load_inline` 的扩展缓存失效，当次 prefill 会付约 80–90 s 的编译时间。**
表现为 250K prefill 从 29 s 跳到 109–111 s（4×），极像性能回退。
规则：改过 `.cu` 之后，**先跑一个 4K 的短 prefill 把编译付掉**，再测长文；
或者至少跑两次 250K，只认第二次。

## decode 并发指标（2026-11-05 新增 · 用户指令"要增加一个测速指标：8 并发、16 并发 decode"）

单一入口：`sm70_tests/conc_decode_metric.sh <tag> <salt-base>`，内部对 `conc ∈ {8,16}` 各跑
`qsa_conc_tput.py --mode decode --conc C --rounds 3`（首轮 warm 不计），salt = base+conc，
**禁用 9502**（那是 prefill 尺子）。取 `decode_agg` 与 `steady_decode`。判读仍走同期控制 +
取轮间 min（定律④⑥）；这一档与 65K ruler 互补：ruler 看单流延迟，本指标看**权重复用**。

首测（TP1、max-num-batched-tokens=8192、同一晚连续测，单位 tok/s 聚合）：

| max-num-seqs | conc=8 | conc=16 | 10K ITL | 65K ITL | canary |
|---|---|---|---|---|---|
| 16（旧基线） | 266.3 / 267.8 | 342.3 / 346.3 | 9.725 ms | 11.000 ms | 7/7 |
| **32（采纳）** | **276.1 / 274.7** | 343.8 / 347.1 | **9.634 ms** | **10.898 ms** | 7/7 |
| 64 | 276.6 | 未出数 | 崩 | — | — |

结论：① **并发本身就是杠杆** —— conc16 比 conc8 高 **+29%**（344 vs 267），因为 decode 是权重流式、
多路复用同一份权重；这是全战役第一次把 decode 聚合吞吐推上去（此前只有 ITL −3.7%，MoE decode 从未变快）。
② `max-num-seqs` 16→32 在 **conc8 +3.1%**、conc16 中性（344→345，噪声内）、prefill_agg 中性
（19061→19055），ITL −0.9% —— 收益来自**调度准入**（8 路解码时仍能准入新 prefill），不是算术变化，
故属用户裁定的"可调参"范围；已采纳为基线（drop-in 加 `Environment=VLLM_MAX_SEQS=32`）。
③ **内存预算的硬顶**：`max-num-seqs=64` 在 10K ITL 腿 OOM（`515899392 B` 分配失败），
`max-num-batched-tokens=16384` 启动即 OOM（`264.00 MiB`）⇒ 两者**不得作为默认**；
TP1 单卡（18 GB 预载权重 + KV）只容得下 seqs≤32 @ batched 8192。
