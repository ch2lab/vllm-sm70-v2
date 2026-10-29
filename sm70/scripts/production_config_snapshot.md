# SM70 生产配置快照（2026-10-29 收尾态）

这份文件是 `/root/run/run_sm70_serve_ipc.sh`（repo 之外）的**受版本控制副本**，
因为 `sm70/*.md` 与 `sm70_tests/*` 都被 .gitignore，一旦换机或误改，调好的配置就会丢。
**改这里之前先确认服务脚本，改服务脚本之后同步回来。**

## 启动参数（逐字）

```
  --load-format ipc_cache
  --kv-cache-dtype nvfp4
  --tensor-parallel-size 8
  --enable-expert-parallel
  --enable-hc-sp
  --max-model-len 524288
  --max-num-batched-tokens 8192
  --max-num-seqs 40
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 48 64 80
  --gpu-memory-utilization 0.83
  --kv-cache-memory-bytes 11200000000
  --attention-config '{"indexer_kv_dtype": "int8"}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":true}'
```

| 参数 | 依据 |
|---|---|
| `--max-num-seqs 40` + `--cudagraph-capture-sizes … 48 64 80` | **M = max_num_seqs × (1+k) = 80 必须 ≤ max(capture_sizes)**。capture 只到 32 时 conc>16 整批掉出 CUDA Graph 走 eager，吞吐掉 3.1–3.5×（conc32 343→1072）。改任何 batch 形状都要复核这条。 |
| `--kv-cache-memory-bytes 11.2 GB` | 池 2,613,672 tokens（9.6 GB 时 2,240,069）。65K 下 40 并发刚好装满。自由显存仍余 ~5 GB/card。 |
| `"indexer_kv_dtype": "int8"` | 同为 128 B/行时比 e5m2 精确 4.4×，解码是一次 int→fp16 cast ⇒ 精度与速度双优。decode1 109.1→111.1。 |
| `num_speculative_tokens: 1` | k=2 实测全并发区间 0.70–0.93×（capture 扩到 96 也不是病因）。 |
| `--max-num-batched-tokens 8192` | chunk 10240→16384 会 Triton OOM（W10）。 |
| `--enable-hc-sp` | prefill 开 sequence parallel、decode 走复制路径（`hc_phase`）。 |

## 关键 env（只列影响性能/正确性的）

```
SM70_PAGE4_TP8=1                      # QSA page4 路由（evicted 的 /tmp 文件阀从此而来）
SM70_QSA_PAGE4_BACKEND=local          # page4 走 in-tree stable-ABI ops
SM70_QSA_FMA=1
VLLM_SM70_AR_IMPL=flat                # 13.79 µs/次；hier 23.6（慢 1.7×）
VLLM_SM70_AR_FLAT_S=8                 # S=1 是 16.99，S=16 是 13.47 ⇒ 8/16 等价，取 8
VLLM_SM70_AR_FLAT_MAX_BYTES=1048576
VLLM_SM70_AR_HIER=1 / AR_WIRE=3       # flat 生效时这两个只影响非 flat 路径
VLLM_SM70_GDN_CHUNKED=1               # GDN prefill 的 K1/K2/K3 链
VLLM_SM70_GREEDY_VERIFY=1             # temperature=0 走 sm70_greedy_verify
VLLM_SM70_MOE_QPN2_NS=1 / QPN2_WIDE=256
VLLM_SM70_MOE_FUSE_ACT=1
VLLM_SM70_PLE_SYNC=1                  # PLE UVA lookup 的 side-stream race（P0 修复）
VLLM_SM70_QSA_STAGES=3 / QSA_VERIFY_PAGE_CAP=512
VLLM_SM70_QSA_PAGE4_SORT=inline32 / QSA_PLAN_PACK=1 / QSA_POS_FUSE=1 / QSA_DECODE_FUSE=0
VLLM_SM70_QWEN38_FP16_GEMV=1
VLLM_SM70_QWEN38_FUSED_HC_UP_GATE=1 / FUSED_HC_DOWN_SILU=1
VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=0   # =1 会 boot 失败（z.shape != x_shape_og）
VLLM_SM70_STEP_SPLIT=1                # 写 /tmp/sm70_step_split_<pid>.jsonl 的 target/draft 毫秒
VLLM_SM70_QSA_SCORE_DIAG=1            # 一次性打印 indexer 打分的 rows/columns/字节
VLLM_SM70_TRITON_TOPK=1
VLLM_SM70_ALLOW_DUP_LAYER=1
```

注意脚本里有重复 export（`GREEDY_VERIFY` / `PLE_SYNC` / `STEP_SPLIT` 各两次），无害但可清理。

## 回归门（改动前后都必须复核）

| 指标 | 当前值 | 测法 |
|---|---|---|
| prefill busy（65K，8 chunk） | **6.707 s** | `sm70_tests/profile_prefill.py` + 按 98 次 qpn2 分段 |
| decode1（65K） | **~110** | `qsa_conc_tput.py --mode decode --len 65536 --conc 1 --salt-seed 9503 --rounds 6` |
| conc40（65K） | **1107–1115 agg** | 同上 `--conc 40` |
| canary | **9/9** | `sm70_tests/correctness.py` |
| 新 Xid | **0** | `journalctl -k --since today \| grep -c "NVRM: Xid"` |
| KV 池 | **2,613,672** | boot 日志 `GPU KV cache size` |

## 今天采纳的代码改动（`git log`）

```
bfa1f75953 perf(gdn): register-prefetch K2's u slice, dropping its smem stage
aeec96dd03 perf(gdn): compute the K3 decay matrix once per (chunk, v-head)
7682a7d80f perf(qsa): fp16 indexer score tile and a 512-wide head-sum block
7c8347e65f refactor(sm70): drive the QSA page4 valves from env, retire the /tmp files
81ffaceeb4 feat(sm70): audit NCCL_* and SM70_* names in the model process
9de5fd056e docs(agents): correct the ipc weight-fingerprint description
8823d335ca perf(hc): tile the SM70 HC up-projection over columns, not one per program
```

## 已知不可回退的坑（都实测过）

- `VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1` → boot 失败（`assert z.shape == x_shape_og`，`_fla_rms_norm_gated`）
- `VLLM_SM70_QSA_VERIFY_PAGE_BUDGET` > 0 → planner 快 19% 但 attention 慢 4.1×（散页破坏 locality），prefill +6.1%
- `AR_IMPL=hier` → 23.6 µs/次（flat 13.79）
- 任何 capture 表未覆盖的 M → 整批 eager（3.1–3.5× 损失）；M 被补齐到下一档时只是浪费（~5%）
