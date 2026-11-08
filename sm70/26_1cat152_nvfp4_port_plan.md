# 1cat 1.5.2 × NVFP4 PLE / NVFP4 KV 移植方案（Flash-Next）

状态：**方案待评审，未动代码**。日期 2026-11-08。

## 0. 基线与事实核验

- **正确源码位置**：`/data/src/1cat-vllm-152`（tag `v1.5.2`，commit `2bb8c2b9cd`，2026-11-06）。
  证据：该树与 `1cat_vllm-1.5.2+cu128` wheel 抽样 60/60 个 `.py` 哈希一致，`ple_layer.py`
  哈希同为 `b4499f55`。
- 排除项：`/data/vllm-1cat/src` 是 **v1.5.1**（分离头指针，2026-10-03）；已 `git checkout` 复原，未留改动。
- 运行环境：`/root/vllm-1cat`（torch 2.10.0+cu128，`flash_qla`、`flash_attn_v100` 均可导入），
  **已安装包与 wheel 完全一致**。运行期修改必须来自源码树构建，不能只改 `site-packages`。
- 当前阻塞（实测）：Flash-Next 在 1cat 上启动时报
  `NotImplementedError: Qwen4Exp pinned-host PLE requires FP8 checkpoint storage`
  （`ple_layer.py:627` 断言，1.5.2 版本同）。

### 检查点的真实格式（Flash-Next-FP4-Local-NVFP4）

| 项 | 值 |
|---|---|
| `ple_embedding_dtype` | `nvfp4` |
| PLE 行存储 | `shard_N.weight` = **U8 [2500012, 80]**（E2M1 码，dim=160/分片） |
| PLE 组尺度 | `shard_N.weight_scale` = **F8_E4M3 [2500012, 10]**（每 16 元素一组） |
| PLE 全局尺度 | `ngram_embedding.weight_scale_2` = **F32 3.324e-05**（每表一个） |
| 分片数 | 128（16 维分片 × 8 词表分片），nvfp4 合计 ≈ 25.6 GB |

### 我们（vllm-v2，已验证生产）的做法 —— 移植参照

- **原生存储**：`codes(U8, dim/2)` + `scales(U8, dim/16)`，**不做 FP8 转换**；
- 取行时用 Triton `_lookup_ple_embedding_nvfp4_kernel` 解码为
  `e2m1(code) × e4m3(group)`，**全局尺度由 `weight_scale` 在 dequantize 时乘**；
- 相关文件：`vllm/models/qwen4_exp/nvidia/nvfp4_ple.py`（LUT/解码 helper）、
  `vllm/models/qwen4_exp/common/ngram_embedding.py`（方法类与取行内核）。

### 1cat 1.5.2 的对应结构

- PLE 运行时全部在 `vllm/models/qwen4_exp/nvidia/ple_layer.py`：
  - `Qwen4ExpPLEFp8EmbeddingMethod`（FP8 方法，含 `create_weights`/`embedding`/`dequantize`）；
  - `Qwen4ExpPinnedHostEmbedding`（device/host 分层表，`_storage_dim` 决定每行字节数）；
  - `Qwen4ExpPLELayer.load_weights`（分片装载：`shard_N.weight` → `load_shard`/`copy_ple_embedding_shard_`）；
  - 取行是 **Python 注册的 Triton op**：`qwen4_exp_ple_pinned_gather`（`ple_layer.py:3388`）
    → `_gather_ple_fp8_from_pinned_kernel`。
- **PLE 不需要改 CUDA**（取行与装载都在 Python/Triton 层）。
- KV：SM70 的 QSA 走 `torch.ops.vllm_sm70_qsa_device.*`（`csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu`）
  与 `qsa_prep_sm70_out`（`csrc/sm70_turbomind/ops/qsa_prep_sm70.cu`）；
  `qsa.py` 明确只接受 **fp8_e4m3(uint8 存储)/fp16/bf16**，无 nvfp4。
  ⇒ **NVFP4 KV 必须改 CUDA**。

## 1. Workstream A：NVFP4 PLE（纯 Python/Triton，不改 CUDA）

目标：Flash-Next 能在 1cat 上 **以 nvfp4 PLE 启动**（KV 先用 fp16，作为 A 的验收条件）。

| 步骤 | 内容 | 文件 |
|---|---|---|
| A1 | 复制我们的解码 helper（LUT、`nvfp4_decode`、`nvfp4_quantize`） | 新增 `vllm/models/qwen4_exp/nvidia/nvfp4_ple.py` |
| A2 | 新增 `Qwen4ExpPLENvFp4EmbeddingMethod(Qwen4ExpPLEFp8EmbeddingMethod)`：`_storage_dim = dim//2 + dim//16`；`weight` 用 uint8 存储；`weight_scale` 承载**全局尺度**（来自 `weight_scale_2`） | `ple_layer.py` |
| A3 | `load_weights` 增加 nvfp4 分支：把 `shard_N.weight`(U8) 与 `shard_N.weight_scale`(F8 2D) 配对，写入每行 `[codes dim/2 \| scales dim/16]`；形状校验从 `(rows, dim)` 改为 `(rows, dim/2)`/`(rows, dim/16)`；`ngram_embedding.weight_scale_2` → `weight_scale` 参数 | `ple_layer.py` |
| A4 | 新增 `qwen4_exp_ple_pinned_gather_nvfp4`（移植我们的 `_lookup_ple_embedding_nvfp4_kernel`，加 codes/scales 的**行步长**参数以适配打包行），在 `embedding_lookup` 里按方法类型分派 | `ple_layer.py` |
| A5 | 覆盖 device / pinned-host / 远端 worker 三条取行路径（远端 worker 同样按打包行解码） | `ple_layer.py` |
| A6 | 工厂：`ple_embedding_dtype == "nvfp4"` → 返回新方法（当前只认 `float8_e4m3fn`） | `ple_layer.py:545` 调用点 |

**验收（A）**：`Application startup complete` + 32-token 生成语义连贯；
用我们的解码函数做行级对拍（随机 1k 行，fp16 输出 bit-exact 或 rel<1e-3）。

## 2. Workstream B：NVFP4 KV（必须改 CUDA）

两种实现，按风险排序：

### B1（推荐先做）分块反量化（fallback，最稳）

- 缓存仍按 **nvfp4 cell 布局**分配（复用 1cat 已有的
  `nvfp4_kv_cache_full_dim` / `nvfp4_kv_cache_split_views`，`torch_utils.py:464/521`），
  即 `[blocks, 2, block_size, kv_heads, 144]`，每 cell = `K数据128 | K尺度16 | V数据128 | V尺度16`；
- 在 QSA 读取前把**被选中的 KV 页**反量化到一块 fp16 scratch（新增一个 Triton/CUDA
  dequant kernel），然后走现有 fp16 内核；
- 改动点：`qsa.py` 的 KV dtype 准入 + `qsa_prep_sm70_out` 写入侧 + 读取侧 dequant；
- 代价：每 chunk 多一次"选中页"反量化；选择是稀疏的（2051/token），可接受；
- 好处：不动现有 attention 内核主体，数值容易对齐，最快拿到可运行的 nvfp4 KV。

### B2（目标态）内核内反量化

- 把我们的 `sm70_nvfp4_decode*.py` / 对应 CUDA 的 cell 解码直接接进
  `vllm_sm70_qsa_device` 的 `run_rows/run_tiled(+window)` 与 QSA prefill 内核；
- 改动点：`csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu`、`qsa_prep_sm70.cu`，
  以及 `csrc/qsa/prefill*.cu`（QSA prefill）；
- 收益：省掉 scratch 往返（B1 的额外读写），但需要更完整的数值/边界验证。

### 入口与开关

- `qsa.py`：`supported_kv_cache_dtypes` 增加 `"nvfp4"`；`kv_cache_dtype_str_to_dtype("nvfp4") → torch.uint8`；
  解码路径按 `self.kv_cache_dtype == "nvfp4"` 分支。
- 运行时阀门：`VLLM_SM70_QSA_KV_NVFP4=1`（默认关），便于 A/B 与回滚。

## 3. 构建与验证

1. **Python-only 阶段（A）**：可用 `PYTHONPATH=/data/src/1cat-vllm-152` 覆盖到已安装环境验证，
   不触发重编；确认后仍以源码树构建产物为准。
2. **CUDA 阶段（B）**：在 `/root/vllm-1cat` 环境里对源码树构建
   （`pip install -e . --no-build-isolation`，ccache 复用），只重编改动到的 TU。
3. **功能门**：`Application startup complete`（禁 HTTP /health 探测）+ 32-token 生成。
4. **性能门**：我们的 `bench_longctx_sweep.py`（salt 9918、maxlen 262144）与 `qsa_conc_tput.py`，
   记录 8K/64K prefill+TBT、conc8；与 1cat 原生 fp16 KV 对照。
5. **回滚**：每个 workstream 独立阀门（A：`VLLM_PLE_NVFP4=0` 回退 FP8 PLE；B：`VLLM_SM70_QSA_KV_NVFP4=0`）。

## 4. 里程碑与判定

| 里程碑 | 内容 | 判定 |
|---|---|---|
| M1 | A 完成，Flash-Next 以 **nvfp4 PLE + fp16 KV** 启动 | 启动日志 + 生成连贯 + 行级解码对拍 |
| M2 | B1，Flash-Next 以 **nvfp4 KV** 启动（分块反量化） | 同上 + 与 fp16 KV 输出一致性（greedy 文本） |
| M3 | B2，内核内反量化，替换 B1 | M2 全部门 + prefill/decode 不劣于 B1 |

## 5. 风险与预案

| 风险 | 预案 |
|---|---|
| 1cat 的 PLE 分层/offload 路径多（device/host/disk/worker），A4/A5 需覆盖全 | 先只跑 device+pinned-host（关 disk/cascade），其余分支加显式 raise 便于发现 |
| 打包行布局改变 `_storage_dim` 会影响 host budget 计算 | 用 `VLLM_QWEN4EXP_PLE_HOST_GIB` 显式指定，先保证能起 |
| QSA 内核签名与我们的不同，B2 移植量大 | 先做 B1（不改主体内核）拿到可运行 |
| 源码树 10-03 与 1.5.2 的 `.cu` 差异不可知 | 已确认用 `1cat-vllm-152`（v1.5.2 tag）；构建产物与 wheel 的 `.so` 版本对齐由构建日志核验 |
| 构建失败/耗时 | ccache + 只重编改动 TU；失败则退回 PYTHONPATH 覆盖做 A 的验证，B 单独排期 |

## 6. 明确不做

- 不改 `/data/src/vllm-v2`（我们的生产树）；
- 不把 nvfp4 PLE 转成 FP8（偏离我们已验证的存储/解码语义）；
- 不改 1cat 的启动参数默认值（沿用其官方 profile，仅按需加 `--kv-cache-dtype nvfp4`）。
