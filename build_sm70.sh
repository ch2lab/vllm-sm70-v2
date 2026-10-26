#!/bin/bash
# SM70 增量编译脚本
# 用法: ./build_sm70.sh [targets...]   (默认 = 生产 .so 最小集)
#
# 最小化: 只编生产实际用到的 target（事实依据 = 旧树 build-sm70 产物:
#   _C_stable_libtorch + _moe_C_stable_libtorch）。_vllm_fa2_C(FA2, SM80+)
# 等 target 一律不编——SM70 走降级路径, 不依赖它们。
# 增量: ninja + ccache, 只重编改动文件; 换 nvcc 版本才重新 configure。
# 默认离线 + 自动拉取（用户定案）: .deps = 版本化本地依赖缓存, 构建前预检同步:
#   本地版本 == cmake 钉 → 全程零网络（FULLY_DISCONNECTED 兜底保证）;
#   上游 bump 钉 → 自动拉取对应钉定版本（唯一网络事件, 显著日志）。
#
# 不会触发 cu130 覆盖（不使用 uv pip install）。
#
# 与旧版的关键差异（2026-09-29 迁移到 8xV100 主机后必须改）：
#   1. 显式钉住 CUDA 编译器。本主机 apt 里还装着 CUDA 12.0 的 /usr/bin/nvcc，
#      它解析不了 Ubuntu 24.04 glibc 2.39 的 stdlib.h（`_Float32 does not name
#      a type`），而 CMake 不指定时就会选中它 → configure 直接失败。
#   2. 编译器版本变化时强制重新 configure。旧逻辑只在 build.ninja 不存在时
#      configure，换 toolkit 会静默沿用旧 cache（编译出错误 SASS 还不报错）。
#   3. torch 门禁查的是真正重要的不变量：sm_70 kernel 在不在，而不是
#      torch.version.cuda 的字符串。

set -e
cd /data/src/vllm-v2

TARGETS="${1:-_C_stable_libtorch _moe_C_stable_libtorch sm70_flat_ar}"
[ -n "${1:-}" ] && TARGETS="$1"
BUILD_DIR="build-sm70"
JOBS="${JOBS:-32}"

[ -f /data/src/sm70_env.sh ] && source /data/src/sm70_env.sh

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
[ -x "$NVCC" ] || { echo "FATAL: nvcc not found at $NVCC"; exit 1; }
NVCC_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
export CUDACXX="$NVCC"
export PATH="/data/vllm-dev/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"

# deepgemm 用上游官方本地口子（cmake/external_projects/deepgemm.cmake:
# DEEPGEMM_SRC_DIR 设了就直接用该目录, 完全绕开 FetchContent/clone）;
# 版本正确性由下面的 .deps 预检保证。
export DEEPGEMM_SRC_DIR="$PWD/.deps/deepgemm-src"

# ================= .deps 预检同步（默认离线 + 钉变更自动拉取） =================
# 钉值镜像 cmake/external_projects/*.cmake 的 GIT_TAG/_UPSTREAM_TAG/CUTLASS_REVISION:
#   上游 bump 钉时同步更新本表（漂移会被下面的 drift 检查打 FATAL, 不会静默陈旧）。
# 格式: 本地目录名|仓库URL|钉定版本|声明钉值的cmake文件
DEP_MANIFEST=(
  "cutlass-src|https://github.com/nvidia/cutlass.git|v4.7.1|CMakeLists.txt"
  "deepgemm-src|https://github.com/vllm-project/DeepGEMM.git|727f847082d2e7946f70dc02b94949db206310d3|cmake/external_projects/deepgemm.cmake"
  "flashkda-src|https://github.com/vllm-project/FlashKDA.git|17a037d98da546deb4591e967cf961a43c034d8b|cmake/external_projects/flashkda.cmake"
  "flashmla-src|https://github.com/vllm-project/FlashMLA|5dceadbd9eabc9acfc547ef18ec8f7b14b980682|cmake/external_projects/flashmla.cmake"
  "fmha_sm100-src|https://github.com/vllm-project/MSA|968061ab16cb14991a5c7d4c0e8eabf531be0dcf|cmake/external_projects/fmha_sm100.cmake"
  "qutlass-src|https://github.com/IST-DASLab/qutlass.git|e74319e3405ce6d71965732880f5dc1f52371f64|cmake/external_projects/qutlass.cmake"
  "tml_fa4-src|https://github.com/vllm-project/tml-fa4|75765e76a9c2c012c1f6ecd64577eb646eb4d303|cmake/external_projects/tml_fa4.cmake"
  "vllm-flash-attn-src|https://github.com/vllm-project/flash-attention|b273a02621d3c94bc747df67204f591b67f38fae|cmake/external_projects/vllm_flash_attn.cmake"
)
# 不纳管（SM70 不构建, 本地无 src, 按需再说）: deepselect, triton_kernels

sync_dep_cache() {
  local entry name url pin cfile src head want
  for entry in "${DEP_MANIFEST[@]}"; do
    IFS='|' read -r name url pin cfile <<<"$entry"
    # 漂移检查: 钉值必须仍在 cmake 文件里（上游 bump 后忘了更新本表 → 硬错）
    if ! grep -qF "$pin" "$cfile"; then
      echo "FATAL: $name 钉 $pin 不在 $cfile 中 — 上游钉已变更, 更新 DEP_MANIFEST 后重跑"
      exit 1
    fi
    src="$DEPS_DIR/$name"
    if [ -d "$src/.git" ]; then
      head=$(git -C "$src" rev-parse HEAD 2>/dev/null)
      want=$(git -C "$src" rev-parse "$pin" 2>/dev/null || true)
      if [ -n "$want" ] && [ "$head" = "$want" ]; then
        echo "[deps] $name @ $pin (本地, 离线)"
        continue
      fi
      echo "=== [deps] $name 版本更新 (本地 ${head:0:8} -> 钉定 $pin): 自动拉取 ==="
      git -C "$src" fetch --quiet --tags origin || true
      want=$(git -C "$src" rev-parse "$pin" 2>/dev/null || true)
      if [ -z "$want" ]; then
        echo "=== [deps] $name 钉不在本地历史(浅克隆): 重新克隆 ==="
        rm -rf "$src"
        git clone --quiet --recurse-submodules "$url" "$src"
        git -C "$src" checkout -q "$pin"
        continue
      fi
      git -C "$src" checkout -q "$pin"
      git -C "$src" submodule update --init --quiet --recursive
    else
      echo "=== [deps] $name 无本地缓存: 自动拉取 $pin ==="
      rm -rf "$src"
      git clone --quiet --recurse-submodules "$url" "$src"
      git -C "$src" checkout -q "$pin"
    fi
  done
}
DEPS_DIR="$PWD/.deps"
sync_dep_cache


# 记录本 build 目录用的是哪个 nvcc，换版本就重新 configure
STAMP="$BUILD_DIR/.sm70_nvcc_version"
if [ -f "$BUILD_DIR/build.ninja" ] && [ "$(cat "$STAMP" 2>/dev/null)" != "$NVCC_VER" ]; then
    echo "=== CUDA toolkit changed ($(cat "$STAMP") -> $NVCC_VER): reconfiguring ==="
    rm -rf "$BUILD_DIR"
fi

if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "=== First run: configuring cmake (nvcc $NVCC_VER) ==="
    cmake -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$NVCC" \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DVLLM_TARGET_DEVICE=cuda \
        -DVLLM_PYTHON_EXECUTABLE=/data/vllm-dev/bin/python \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DFETCHCONTENT_BASE_DIR=/data/src/vllm-v2/.deps \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    mkdir -p "$BUILD_DIR"
    echo "$NVCC_VER" > "$STAMP"
fi

for T in $TARGETS; do
  echo "=== Building target: $T (nvcc $NVCC_VER, -j$JOBS) ==="
  # 直接用 ninja 避免 cmake --build 触发重新 configure（需要网络）
  ninja -C "$BUILD_DIR" -j"$JOBS" "$T"
done

# 复制 .so 到 vllm 包目录（支持 .abi3.so 和 .cpython-*.so 两种格式）
for T in $TARGETS; do
  SO_FILE=""
  if [ -f "$BUILD_DIR/${T}.abi3.so" ]; then
    SO_FILE="$BUILD_DIR/${T}.abi3.so"
  elif ls "$BUILD_DIR/${T}".cpython-*.so >/dev/null 2>&1; then
    SO_FILE=$(ls "$BUILD_DIR/${T}".cpython-*.so | head -1)
  elif [ -f "$BUILD_DIR/${T}.so" ]; then
    # Plain (non-extension) shared library, e.g. sm70_flat_ar (ctypes-loaded).
    SO_FILE="$BUILD_DIR/${T}.so"
  fi
  if [ -n "$SO_FILE" ]; then
    cp "$SO_FILE" "vllm/$(basename "$SO_FILE")"
    echo "=== Installed: vllm/$(basename "$SO_FILE") ==="
  else
    echo "WARNING: no .so produced for $T"
  fi
done

# torch 门禁：必须是带 sm_70 kernel 的本地源码编译版
# （官方 wheel 已移除 Volta，见 pytorch/pytorch#157517，装错会静默失去 SM70）
TORCH_OK=$(/data/vllm-dev/bin/python -c "
import torch
print('ok' if 'sm_70' in torch.cuda.get_arch_list()
      and torch.__version__.startswith('2.13.0a0+git') else 'bad')
" 2>/dev/null || echo bad)
if [ "$TORCH_OK" != "ok" ]; then
    echo "WARNING: torch lost its sm_70 kernels, restoring local cu128 build..."
    /data/vllm-dev/bin/pip install \
        /data/src/pytorch-src/dist/torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl \
        --force-reinstall --no-deps -q
fi

echo "=== Done ==="
