# MoE prefill 重写工程方案（Q36 TP1, SM70）

> 状态：**未启动**（本文件是立项输入，不是已实施记录）。所有"已测"数据来自
> `sm70/03_history.md` (55)–(60)，命令与脚本见 `/data/sm70_results/screens/`。

## 1. 为什么值得做：实测缺口

| 参照 | ms（w13, chunk 8192） | TF/s | %峰值(123) |
|---|---|---|---|
| 生产 `nvfp4_qpn2_moe_wide` | 6.686 | 41.2 | 33.5% |
| cuBLAS 分批 GEMM（同调度） | 4.09–4.15 | 66–67 | 54% |
| 本仓 wmma 原型（128 行 tile） | 8.21 | 33.5 | 27.2% |

- MoE 占 **65K 尺子 prefill 的 23.4%**（`q36_prefill65` profile，960 次 × 5.60ms）。
- 兑现 1.63× ⇒ **≈9% 全量 prefill**（65K prefill 17.2s → ~15.7s）。
- 缺口**不是**形状（chunk 8K→32K 平坦 41.1→41.5）、**不是**占用率（强压 2 CTA/SM 慢 9.2×）、
  **不是**反量化（预解量化 B 更慢 24.7 vs 34.5）、**不是**启动器旋钮（≤3.7%）。

## 2. 机制（唯一自洽解释）

生产 CTA 的 M tile = **16 行**（route block）× N=256 列。每个 B 元素只服务 16 行，
mma:load ≈ 2（4 次 mma 换 8 个 B 寄存器）。cuBLAS 用 128 行 tile（复用 8×）才到 54%。
⇒ 需要 **大 M tile + 高效 B 暂存**；内核当前**只支持 8/16 行 route block**，无配置路径。

## 3. 目标设计

- **Tile**：CTA = 128 行（一个专家的全部 token 或 128 行切片）× 128 列；8 warps，每 warp 32×64
  （16 个 m8n8k4 累加器 = 64 float/lane）。
- **权重常驻**：每个 CTA 固定一个专家，B 面板按 K 分块进 smem 并**只读一次**；token 循环在内。
- **Smem**：A/B 双缓冲，K 分块 32；**swizzle**（XOR 行/列）消除 wmma/LDS bank 冲突；
  Volta 无 cp.async ⇒ LDG→寄存器→STS 的手工流水（原型已验证该模式本身不是瓶颈，见 (56)）。
- **mma**：手写 `mma.sync.aligned.m8n8k4`，片段布局**直接复用**
  `csrc/quantization/marlin/sm70_nvfp4_qpn2.cu:192` 的宏与 `local_row/quadpair` 映射；
  B 走 nvfp4 内联反量化（保留 fp4 字节，不走 Fp16B）。
- **K-split**：K=2048 一次到底（无需 split），收尾直接写 out（fuse_act 可选）。

## 4. 验收（硬门槛，缺一不可）

1. **位等**：与 cuBLAS bmm（同形状 fp16 权重）`max|d| = 0`；nvfp4 路径与生产内核逐元素相同。
2. **CUPTI 同进程**：≥ **45.3 TF/s**（生产 41.2 × 1.1）才算过屏；目标 55+。
3. **服务三明治门**（同场）：conc8/conc16 基线前后夹 + 65K/250K prefill + canary 7/7 +
   recall 9/9 + secret 6/6 + **0 新 Xid**。
4. **回滚阀**：`VLLM_SM70_MOE_PREFILL_WMAJOR`（默认关），关时走原内核。

## 5. 风险与工作量

- 片段语义非标准（8 个累加器/4 次 mma 覆盖 16 k），需以生产内核为参考逐位校准；**估计 2–3 天**。
- 本机 **ncu 不可用**（驱动问题），只能靠 CUPTI kernel-time + `cuobjdump -sass` 交叉验证。
- 若 wmma/手写均无法过 45.3，则**记录 NO-GO 并保留生产内核**（不降级现有 41.2）。
