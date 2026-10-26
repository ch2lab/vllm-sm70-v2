
#include <cute/tensor.hpp>

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

// Named (not anonymous) namespace: cutlass mma_traits_sm70.hpp contains its
// own anonymous namespace, and two anonymous namespaces in one nvcc TU make
// the generated host-stub _NV_ANON_NAMESPACE reference ambiguous.
namespace fdhmma {

using namespace cute;

// Head geometry is per-instantiation (template parameters of the kernels and
// host launch): TP2 = (kH, kHQ) = (2, 12); TP4 = (1, 6). kRatio is shared.
constexpr int kRatio = 6;
constexpr int kD = 256;
constexpr int kN = 16;
constexpr int kMaxQ = 8;
constexpr int kDD = kD / 2;
constexpr int kSD = kD / 16;
constexpr int kF = kDD + kSD;
constexpr int kThreads = 128;
constexpr int kPartMaxThreads = kRatio * 32;   // 192
constexpr int kPartWarps = kRatio;
constexpr int kLaneDims = 8;
constexpr int kRowStride = kD + 2;
constexpr int kDChunks = 4;                     // 64-dim chunks
// Merged-verify: for 2 <= q_rows <= kMergedMaxQ ONE CTA computes every
// (qq, head) score as a gemm M-row (q_rows * kRatio <= 24 rows = kMtiles
// tiles of 8), so the KV chunk is staged/dequantized once per split
// instead of once per row (production MTP3 verify = 4x less KV traffic).
constexpr int kMergedMaxQ = 4;
constexpr int kMtiles = 3;                      // ceil(4*6 / 8)

using SmemTileAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

using FmAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
using FmTiled = TiledMMA<FmAtom, Layout<Shape<_1, _2, _1>>,
                         Tile<_8, Int<kN>, _4>>;

// PV gemm P[m,kv] @ V[kv,d] -> O[m,d]: the B-operand (V) must be staged
// d-first [d][kv] (kv contiguous, PLAIN -- the [8,64] swizzle block can't fit a
// 16-wide kv dim). partition_B always maps the B-tile's first index to N and
// second to K, so with N=d, K=kv the tile is [d][kv] at coord (nt, 0). This works
// with either the TN or TT atom; we keep the TN atom (FmTiled, same as QK).
using PvTiled = FmTiled;

__device__ __forceinline__ float e2m1_f(unsigned code) {
  const unsigned m = code & 7;
  float mag;
  if (m < 5) {
    mag = m * 0.5f;
  } else if (m < 7) {
    mag = (m == 5) ? 3.0f : 4.0f;
  } else {
    mag = 6.0f;
  }
  return (code & 8) ? -mag : mag;
}

__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  const bool sgn = b & 0x80;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  float v;
  if (e == 0) {
    v = m * 0.001953125f;
  } else {
    v = (1.0f + m * 0.125f) * exp2f((float)e - 7.0f);
  }
  return sgn ? -v : v;
}

// MODE: 1 = per-row CTA (q_rows=1 or >8); 2 = merged q_rows 2..4 (3 CTAs/SM);
// 3 = merged q_rows 5..8 (48 M-rows: 24KB q_smem -> 2 CTAs/SM).
// PVHMA: run the PV (P@V) gemm on the 8x8x4 TC pipe instead of the FP32
// FFMA pipe (V100's only available mma shape; ~2x the FFMA pipe rate).
// MERGED modes only (host gates); the FFMA path is bit-identical when off.
template <int MODE, int kH, int kHQ, bool PVHMA = false>
__global__ void __launch_bounds__(
    kPartMaxThreads, MODE == 3 ? 2 : (MODE == 2 ? 3 : 4))
sm70_fd_hmma_partition_kernel_impl(
    const __half* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const int* __restrict__ bt,
    const int* __restrict__ kv_lens,
    float* __restrict__ ws,
    float k_scale, float v_scale, float sm_scale,
    int window_left, int splits, int q_rows, int bt_cols, int n_page,
    int64_t page_stride, int64_t side_off) {
  constexpr bool MERGED = (MODE >= 2);
  const int zrows = MERGED ? 1 : q_rows;
  const int r = blockIdx.z / zrows;
  const int qq0 = MERGED ? 0 : blockIdx.z % q_rows;
  const int nrows = MERGED ? q_rows : 1;
  const int h = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // M-rows are (qq, head) pairs flattened as i = qq*kRatio + head; one
  // gemm tile per 8 rows. !MERGED: i == head (rows 6..7 zero pad).
  constexpr int kSmemRows =
      MODE == 1 ? 8 : (MODE == 2 ? kMtiles * 8 : 6 * 8);
  __shared__ __half q_smem[kSmemRows * kD];     // 12 KB merged, 4 KB plain
  // !PVHMA: V packed bytes (2.3 KB). PVHMA: fp16 V tile (8 KB). One union
  // keeps the PVHMA smem at ~47 KB -> still 2 CTAs/SM in MODE3.
  __shared__ union {
    uint8_t kv_raw[kN * kF];
    __half sV_storage[kN * kD];
  } kvu;
  __shared__ __half k_tile_storage_full[kN * kD];   // 8 KB (gemm operand)
  // Score slabs: MERGED = one fully-accumulated slab per M tile (tiles
  // 0..2); !MERGED = one partial per dim chunk for the single tile
  // (softmax sums the 4). Same [kDChunks][8][kN] bytes serve both.
  __shared__ float s_sc[MODE == 3 ? 6 : kDChunks][8][kN];
  // v3h-style dequant tables: e2t[byte] = the 2 nibble magnitudes,
  // e4t[side*256+byte] = e4m3 value x layer scale (exact in fp32).
  __shared__ float2 e2t[256];
  __shared__ float e4t[512];
  // K-dequant nibble table (16 signed e2m1 values): the K path uses this
  // instead of byte-indexed e2t (random float2 reads bank-conflict hard);
  // e2t stays for the V paths (PVHMA staging + FFMA PV).
  __shared__ float t16[16];

  auto sK = make_tensor(make_smem_ptr(
      reinterpret_cast<half_t*>(k_tile_storage_full)),
      tile_to_shape(SmemTileAtom{}, Shape<Int<kN>, Int<kD>>{}));
  // PVHMA: fp16 V tile staged d-first [d][kv] (kv contiguous, PLAIN) so the PV
  // gemm's B-operand reads [N=d, K=kv] at B-tile coord (nt, 0). The [8,64]
  // swizzle can't fit the 16-wide kv dim, so no swizzle (the FFMA path's V is
  // read in place from kv_raw; this fp16 tile is PVHMA-only).
  auto sV = make_tensor(
      make_smem_ptr(reinterpret_cast<half_t*>(kvu.sV_storage)),
      make_shape(Int<kD>{}, Int<kN>{}), make_stride(Int<kN>{}, _1{}));
  __shared__ float s_alpha[kPartWarps][8];   // PVHMA per-row alpha
  __shared__ float s_l_row[kPartWarps][8];  // PVHMA per-row l (diverges/lane)

  const int kv_len = __ldg(kv_lens + r);
  const int64_t ws_sp = (int64_t)q_rows * kRatio * kRowStride;
  const int64_t ws_base =
      ((int64_t)(r * kH + h) * splits + sp) * ws_sp;

  float* slot = ws + ws_base + qq0 * (kRatio * kRowStride);
  auto write_empty = [&]() {
    for (int idx = tid; idx < nrows * kRatio * kRowStride;
         idx += kPartMaxThreads) {
      const int rem = idx % kRowStride;
      slot[idx] = (rem == kD) ? -INFINITY
                : (rem == kD + 1) ? 0.0f : 0.0f;
    }
  };

  int lo = window_left >= 0 ? kv_len - 1 - window_left : 0;
  if (lo < 0) lo = 0;
  write_empty();
  if (kv_len == 0 || lo >= kv_len) return;
  const int64_t t_a = (int64_t)lo + ((int64_t)kv_len - lo) * sp / splits;
  const int64_t t_b =
      (int64_t)lo + ((int64_t)kv_len - lo) * (sp + 1) / splits;
  if (t_a >= t_b) return;
  const int c0 = (int)(t_a / kN);
  const int c1 = (int)((t_b - 1) / kN);
  constexpr int KNR = MERGED ? (MODE == 2 ? kMergedMaxQ : kMaxQ) : 1;
  int hi_row[KNR];
#pragma unroll
  for (int qi = 0; qi < KNR; qi++) {
    hi_row[qi] = (qi < nrows) ? kv_len - q_rows + qq0 + qi + 1 : 0;
  }

  // ---- Q into smem once (all owned rows; zero pads, sm_scale folded) ----
  {
    const int nq = nrows * kRatio;
    const __half* src =
        q + ((int64_t)(r * q_rows + qq0) * kHQ + h * kRatio) * kD;
    for (int idx = tid; idx < kSmemRows * kD / 2;
         idx += kPartMaxThreads) {
      const int t = idx / (kD / 2);
      const int d2 = (idx % (kD / 2)) * 2;
      const __half2 s = __float2half2_rn(sm_scale);
      const __half2 v = (t < nq)
          ? *reinterpret_cast<const __half2*>(
                src + (t / kRatio) * (kHQ * kD) + (t % kRatio) * kD + d2)
          : __half2();
      reinterpret_cast<__half2*>(&q_smem[0])[idx] = __hmul2(v, s);
    }
    for (int b = tid; b < 256; b += kPartMaxThreads) {
      e2t[b] = make_float2(e2m1_f(b & 0xF), e2m1_f(b >> 4));
      const float sfv = e4m3_to_float(static_cast<unsigned>(b));
      e4t[b] = sfv * k_scale;
      e4t[256 + b] = sfv * v_scale;
    }
    for (int b = tid; b < 16; b += kPartMaxThreads) t16[b] = e2m1_f(b);
  }
  __syncthreads();

  // ---- gemm setup: A tile t = q_smem rows 8t..8t+7 ----
  FmTiled tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(lane);
  PvTiled pv_tiled;
  auto pv_thr = pv_tiled.get_thread_slice(lane);
  const int d_base = kLaneDims * lane;
  float acc[KNR][kLaneDims];          // !PVHMA PV accumulator
  float m_row[8];                      // PVHMA per-row m (tile rows)
  float alpha_row[8];                  // PVHMA per-row alpha this chunk
  float m_run[KNR], l_run[KNR];        // !PVHMA per-(head, qi) m/l
  float alpha_q[KNR];
#pragma unroll
  for (int qi = 0; qi < KNR; qi++) {
    m_run[qi] = -INFINITY;
    l_run[qi] = 0.0f;
    alpha_q[qi] = 1.0f;
    if constexpr (!PVHMA) {
#pragma unroll
      for (int j = 0; j < kLaneDims; j++) acc[qi][j] = 0.0f;
    }
  }
  if constexpr (PVHMA) {
    if (warp < (nrows * kRatio + 7) / 8) {
      for (int i = 0; i < 8; i++) {
        m_row[i] = -INFINITY;
        s_l_row[warp][i] = 0.0f;
        alpha_row[i] = 1.0f;
      }
    }
  }
  // PVHMA PV accumulators: one [8M x 16N] C-fragment per d-tile (16 tiles =
  // the 256-dim output row; 64 f32/thread, same budget as the FFMA acc).
  auto pacc2_ref = make_tensor(
      make_gmem_ptr(slot), make_shape(_8{}, Int<kN>{}),
      make_stride(Int<kN>{}, _1{}));
  auto pacc2_proto =
      pv_thr.make_fragment_C(pv_thr.partition_C(pacc2_ref));
  using PVAcc = decltype(pacc2_proto);
  PVAcc pacc2[kD / 16];
  if constexpr (PVHMA) {
#pragma unroll
    for (int nt = 0; nt < kD / 16; nt++) clear(pacc2[nt]);
  }

  for (int c = c0; c <= c1; c++) {
    const int cbase = c * kN;
    const int blk = cbase / n_page;
    const int n0_in = cbase - blk * n_page;
    const int64_t base =
        (int64_t)__ldg(bt + r * bt_cols + blk) * page_stride;
    // ---- V: packed bytes -> smem (FFMA PV reads it in place below);
    // PVHMA skips this entirely (V dequants straight to the fp16 tile) ----
    if constexpr (!PVHMA) {
    for (int cc = tid; cc < kN * (kF / 16); cc += kPartMaxThreads) {
      const int n = cc / (kF / 16);
      const int f16 = cc % (kF / 16);
      const int ntok = n0_in + n;
      int64_t off;
      if (f16 < kDD / 16) {
        off = (int64_t)h * (n_page * kDD) + ntok * kDD + f16 * 16;
      } else {
        off = (int64_t)kH * (n_page * kDD) +
              (int64_t)h * (n_page * kSD) + ntok * kSD +
              (f16 - kDD / 16) * 16;
      }
      const uint4 src = *reinterpret_cast<const uint4*>(
          kv + base + side_off + off);
      *reinterpret_cast<uint4*>(&kvu.kv_raw[n * kF + f16 * 16]) = src;
    }
    }
    // ---- K: global -> registers -> fp16 gemm tile (uint4 of packed
    // bytes = 32 dims = two 16-dim scale groups) ----
    for (int e = tid; e < kN * 8; e += kPartMaxThreads) {
      const int n = e >> 3;
      const int u = e & 7;
      const int ntok = n0_in + n;
      const uint4 d4 = *reinterpret_cast<const uint4*>(
          kv + base + (int64_t)h * (n_page * kDD) + ntok * kDD + u * 16);
      const uint8_t* sc =
          kv + base + (int64_t)kH * (n_page * kDD) +
          (int64_t)h * (n_page * kSD) + ntok * kSD;
      const float sfs[2] = {e4t[sc[2 * u]], e4t[sc[2 * u + 1]]};
      const unsigned words[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
      for (int w4 = 0; w4 < 4; w4++) {
        const float sf = sfs[w4 >> 1];
        const unsigned wd = words[w4];
        // 16-entry nibble table: bit-identical to e2t[byte] per nibble,
        // without the random float2 bank conflicts of the byte-indexed path.
        const unsigned nib[8] = {wd & 0xF, (wd >> 4) & 0xF,
                                  (wd >> 8) & 0xF, (wd >> 12) & 0xF,
                                  (wd >> 16) & 0xF, (wd >> 20) & 0xF,
                                  (wd >> 24) & 0xF, wd >> 28};
        const __half2 hv[4] = {
            __floats2half2_rn(t16[nib[0]] * sf, t16[nib[1]] * sf),
            __floats2half2_rn(t16[nib[2]] * sf, t16[nib[3]] * sf),
            __floats2half2_rn(t16[nib[4]] * sf, t16[nib[5]] * sf),
            __floats2half2_rn(t16[nib[6]] * sf, t16[nib[7]] * sf)};
        *reinterpret_cast<uint4*>(&sK(n, 32 * u + 8 * w4)) =
            *reinterpret_cast<const uint4*>(hv);
      }
    }
    if constexpr (PVHMA) {
      // V: dequant straight to the transposed fp16 tile (shared, __shared__
      // union). V codes on the V side (base + side_off); 16 scale bytes/token.
      for (int e = tid; e < kN * 8; e += kPartMaxThreads) {
        const int n = e >> 3;
        const int u = e & 7;
        const int ntok = n0_in + n;
        const uint4 d4v = *reinterpret_cast<const uint4*>(
            kv + base + side_off + (int64_t)h * (n_page * kDD) +
            ntok * kDD + u * 16);
        const uint8_t* scv =
            kv + base + side_off + (int64_t)kH * (n_page * kDD) +
            (int64_t)h * (n_page * kSD) + ntok * kSD;
        const float sfv[2] = {e4t[256 + scv[2 * u]],
                              e4t[256 + scv[2 * u + 1]]};
        const unsigned wv[4] = {d4v.x, d4v.y, d4v.z, d4v.w};
#pragma unroll
        for (int w4 = 0; w4 < 4; w4++) {
          const float sf = sfv[w4 >> 1];
          const unsigned wd = wv[w4];
          const float2 n01 = e2t[wd & 0xFF], n23 = e2t[(wd >> 8) & 0xFF];
          const float2 n45 = e2t[(wd >> 16) & 0xFF], n67 = e2t[wd >> 24];
          const __half2 hvv[4] = {
              __floats2half2_rn(n01.x * sf, n01.y * sf),
              __floats2half2_rn(n23.x * sf, n23.y * sf),
              __floats2half2_rn(n45.x * sf, n45.y * sf),
              __floats2half2_rn(n67.x * sf, n67.y * sf)};
          // B-operand is [N=d, K=kv]: stage V d-first, sV[d][kv] = V[kv, d]
          // (kv contiguous, plain). col=d, n=kv.
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int col = 32 * u + 8 * w4 + 2 * j;
            sV(col, n) = half_t(__low2half(hvv[j]));
            sV(col + 1, n) = half_t(__high2half(hvv[j]));
          }
        }
      }
    }
    __syncthreads();
    // ---- score gemm ----
    if constexpr (MERGED) {
      // warp g < kMtiles: M tile g over all dim chunks (accumulated in
      // registers, one full slab published).
      if (warp < (nrows * kRatio + 7) / 8) {
        auto accT = thr_mma.partition_C(make_tensor(
            make_gmem_ptr(&s_sc[warp][0][0]),
            make_shape(_8{}, Int<kN>{}), make_stride(Int<kN>{}, _1{})));
        auto rAcc = thr_mma.make_fragment_C(accT);
        clear(rAcc);
#pragma unroll
        for (int dc = 0; dc < kDChunks; dc++) {
          auto sKc = cute::local_tile(sK, make_shape(_16{}, _64{}),
                                      make_coord(0, dc));
          auto gB = thr_mma.partition_B(sKc);
          auto rK = thr_mma.make_fragment_B(gB);
          copy(gB, rK);
          auto mA = make_tensor(
              make_gmem_ptr(reinterpret_cast<const half_t*>(
                  &q_smem[warp * 8 * kD + dc * 64])),
              make_shape(_8{}, _64{}), make_stride(Int<kD>{}, _1{}));
          auto rQ = thr_mma.make_fragment_A(thr_mma.partition_A(mA));
          copy(thr_mma.partition_A(mA), rQ);
          gemm(tiled_mma, rQ, rK, rAcc);
        }
        copy(rAcc, accT);
      }
    } else {
      // single row: warp g < kDChunks computes dim-chunk g as a partial
      // slab over the kRatio head rows.
      if (warp < kDChunks) {
        auto accT = thr_mma.partition_C(make_tensor(
            make_gmem_ptr(&s_sc[warp][0][0]),
            make_shape(_8{}, Int<kN>{}), make_stride(Int<kN>{}, _1{})));
        auto rAcc = thr_mma.make_fragment_C(accT);
        clear(rAcc);
        auto sKc = cute::local_tile(sK, make_shape(_16{}, _64{}),
                                    make_coord(0, warp));
        auto gB = thr_mma.partition_B(sKc);
        auto rK = thr_mma.make_fragment_B(gB);
        copy(gB, rK);
        auto mA = make_tensor(
            make_gmem_ptr(reinterpret_cast<const half_t*>(
                &q_smem[0] + warp * 64)),
            make_shape(_8{}, _64{}), make_stride(Int<kD>{}, _1{}));
        auto rQ = thr_mma.make_fragment_A(thr_mma.partition_A(mA));
        copy(thr_mma.partition_A(mA), rQ);
        gemm(tiled_mma, rQ, rK, rAcc);
        copy(rAcc, accT);
      }
    }
    __syncthreads();
    // ---- softmax + PV: warp = head; each row qq shares this CTA's KV ----
    const int rt0 = cbase > (int)t_a ? 0 : (int)(t_a - cbase);
    const int rt1p = cbase + kN < (int)t_b ? kN : (int)(t_b - cbase);
    int rt1_q[KNR];
    float m_new_q[KNR];
    int rt1_row[8];
    float m_new_row[8];
    int rt1w = rt0;
    if constexpr (!PVHMA) {
#pragma unroll
      for (int qi = 0; qi < KNR; qi++) {
        int rt1 = rt1p;
        if (qi < nrows) {
          const int hh = hi_row[qi] - cbase;
          if (rt1 > hh) rt1 = hh;
          if (rt1 > kN) rt1 = kN;
          if (rt1 < rt0) rt1 = rt0;
          if (rt1 > rt1w) rt1w = rt1;
          if (rt1 > rt0) {
            const int irow = qi * kRatio + warp;
            const float* sc = &s_sc[MERGED ? irow / 8 : 0][irow % 8][0];
            float m_tile = -INFINITY;
            for (int t = rt0; t < rt1; t++) {
              float s = sc[t];
              if constexpr (!MERGED) {
                s = (s_sc[0][irow][t] + s_sc[1][irow][t]) +
                    (s_sc[2][irow][t] + s_sc[3][irow][t]);
              }
              m_tile = fmaxf(m_tile, s);
            }
            const float mn = fmaxf(m_run[qi], m_tile);
            const float alpha = (m_run[qi] == -INFINITY) ? 0.0f
                              : expf(m_run[qi] - mn);
            l_run[qi] *= alpha;
#pragma unroll
            for (int j = 0; j < kLaneDims; j++) acc[qi][j] *= alpha;
            m_run[qi] = mn;
            m_new_q[qi] = mn;
          }
        }
        rt1_q[qi] = rt1;
      }
    } else {
      // PV warps own M-tiles (8 rows), not heads: per-row m/l state.
      for (int i = 0; i < 8; i++) {
        const int irow = warp * 8 + i;
        int rt1 = rt1p;
        if (irow < nrows * kRatio) {
          const int hh = hi_row[irow / kRatio] - cbase;
          if (rt1 > hh) rt1 = hh;
        } else {
          rt1 = rt0;
        }
        if (rt1 > kN) rt1 = kN;
        if (rt1 < rt0) rt1 = rt0;
        rt1_row[i] = rt1;
        if (rt1 > rt1w) rt1w = rt1;
        if (rt1 > rt0) {
          float m_tile = -INFINITY;
          for (int t = rt0; t < rt1; t++)
            m_tile = fmaxf(m_tile, s_sc[warp][i][t]);
          const float mn = fmaxf(m_row[i], m_tile);
          const float alpha = (m_row[i] == -INFINITY) ? 0.0f
                            : expf(m_row[i] - mn);
          s_l_row[warp][i] *= alpha;
          m_row[i] = mn;
          m_new_row[i] = mn;
          alpha_row[i] = alpha;
        } else {
          m_new_row[i] = 0.0f;   // guarded use only (rt1_row == rt0)
          alpha_row[i] = 1.0f;   // no new max -> no rescale
        }
      }
    }
    if (rt1w > rt0) {
      if constexpr (PVHMA) {
        // ---- P staging: fp16 [8 M-rows x 16 kv] into this warp's slab
        // bytes (the fp32 scores are no longer read after the m/l pass);
        // rows without valid kv in this chunk get 0. ----
        __half* sP = reinterpret_cast<__half*>(&s_sc[warp][0][0]);
        if (warp < (nrows * kRatio + 7) / 8) {
          if (lane < 8) {
            const int irow = warp * 8 + lane;
            const bool ok = irow < nrows * kRatio;
            const float mn = ok ? m_new_row[lane] : 0.0f;
            s_alpha[warp][lane] = ok ? alpha_row[lane] : 0.0f;
            float l_part = 0.0f;
            // sP ALIASES the s_sc[warp] slab (fp16 reinterprets the fp32
            // scores). Read every score into regs BEFORE the fp16 stores: an
            // interleaved read-then-store would let one lane's sP write clobber
            // a neighbour row's score the same lockstep is still reading.
            float pvals[kN];
#pragma unroll
            for (int t = 0; t < kN; t++) {
              float p = 0.0f;
              if (ok && t >= rt0 && t < rt1_row[lane]) {
                p = expf(s_sc[warp][lane][t] - mn);
              }
              pvals[t] = p;
              l_part += p;
            }
#pragma unroll
            for (int t = 0; t < kN; t++)
              sP[lane * kN + t] = __float2half_rn(pvals[t]);
            s_l_row[warp][lane] += l_part;
          }
          __syncwarp();
          // ---- PV: P[8m x 16kv] @ V[16kv x 256d] on the TC pipe ----
          auto aT = make_tensor(make_smem_ptr(sP),
                                make_shape(_8{}, Int<kN>{}),
                                make_stride(Int<kN>{}, _1{}));
          auto aT_p = pv_thr.partition_A(aT);
          auto rA = pv_thr.make_fragment_A(aT_p);
          copy(aT_p, rA);
          // per-row alpha as a C-fragment (stride-(1,0) row broadcast)
          auto aTa = make_tensor(make_smem_ptr(&s_alpha[warp][0]),
                                 make_shape(_8{}, Int<kN>{}),
                                 make_stride(_1{}, _0{}));
          auto aTa_p = pv_thr.partition_C(aTa);
          auto rAl = pv_thr.make_fragment_C(aTa_p);
          copy(aTa_p, rAl);
#pragma unroll
          for (int nt = 0; nt < kD / 16; nt++) {
            auto bT = pv_thr.partition_B(
                cute::local_tile(sV, make_shape(_16{}, _16{}),
                                 make_coord(nt, 0)));
            auto rB = pv_thr.make_fragment_B(bT);
            copy(bT, rB);
            // per-row alpha rescale (positional: same frag layout both)
            float* dacc = pacc2[nt].data();
            const float* dalpha = rAl.data();
#pragma unroll
            for (int j = 0; j < (int)pacc2[nt].size(); j++)
              dacc[j] *= dalpha[j];
            gemm(pv_tiled, rA, rB, pacc2[nt]);
          }
        }
      } else {
      for (int t = rt0; t < rt1w; t++) {
        // one V byte read shared by all rows of this token
        const uint32_t vb4 = *reinterpret_cast<const uint32_t*>(
            &kvu.kv_raw[t * kF + (d_base >> 1)]);
        const float sf = e4t[256 + kvu.kv_raw[t * kF + kDD + (d_base >> 4)]];
        const float2 v01 = e2t[vb4 & 0xFF], v23 = e2t[(vb4 >> 8) & 0xFF];
        const float2 v45 = e2t[(vb4 >> 16) & 0xFF], v67 = e2t[vb4 >> 24];
#pragma unroll
        for (int qi = 0; qi < KNR; qi++) {
          if (qi < nrows && t < rt1_q[qi]) {
            const int irow = qi * kRatio + warp;
            const float* sc = &s_sc[MERGED ? irow / 8 : 0][irow % 8][0];
            float s = sc[t];
            if constexpr (!MERGED) {
              s = (s_sc[0][irow][t] + s_sc[1][irow][t]) +
                  (s_sc[2][irow][t] + s_sc[3][irow][t]);
            }
            const float pe = expf(s - m_new_q[qi]);
            l_run[qi] += pe;
            const float p = pe * sf;
            acc[qi][0] = fmaf(p, v01.x, acc[qi][0]);
            acc[qi][1] = fmaf(p, v01.y, acc[qi][1]);
            acc[qi][2] = fmaf(p, v23.x, acc[qi][2]);
            acc[qi][3] = fmaf(p, v23.y, acc[qi][3]);
            acc[qi][4] = fmaf(p, v45.x, acc[qi][4]);
            acc[qi][5] = fmaf(p, v45.y, acc[qi][5]);
            acc[qi][6] = fmaf(p, v67.x, acc[qi][6]);
            acc[qi][7] = fmaf(p, v67.y, acc[qi][7]);
          }
        }
      }
      }
    }
    __syncthreads();   // slabs/tiles reused next chunk
  }

  // ---- direct write of this split's partials (one slot per row) ----
  if constexpr (!PVHMA) {
#pragma unroll
    for (int qi = 0; qi < KNR; qi++) {
      if (qi < nrows) {
        float* rp = slot + qi * (kRatio * kRowStride) +
                    warp * kRowStride + d_base;
#pragma unroll
        for (int j = 0; j < kLaneDims; j++) rp[j] = acc[qi][j];
        if (lane == 0) {
          float* rs = rp - d_base;
          rs[kD] = m_run[qi];
          rs[kD + 1] = l_run[qi];
        }
      }
    }
  } else {
    // PVHMA: warp owns M-tile warp (8 rows); C-frags -> ws slots.
    if (warp < (nrows * kRatio + 7) / 8) {
#pragma unroll
      for (int nt = 0; nt < kD / 16; nt++) {
        auto oT = make_tensor(
            make_gmem_ptr(slot + (warp * 8) * kRowStride + nt * 16),
            make_shape(_8{}, _16{}), make_stride(Int<kRowStride>{}, _1{}));
        copy(pacc2[nt], pv_thr.partition_C(oT));
      }
      if (lane < 8) {
        const int irow = warp * 8 + lane;
        if (irow < nrows * kRatio) {
          slot[irow * kRowStride + kD] = m_row[lane];
          slot[irow * kRowStride + kD + 1] = s_l_row[warp][lane];
        }
      }
    }
  }
}

// Two-stage LSE reduce: stage 1 splits the O(splits) read loop across
// kSplitChunks more CTAs (24 CTAs at R=1 latency-bound the whole reduce at
// ~40GB/s effective; 4x CTAs -> ~4x), emitting per-chunk (m, l, acc)
// triples; stage 2 merges the kSplitChunks triples (4 iterations) into out.
constexpr int kSplitChunks = 4;

template <int kH, int kHQ>
__global__ void __launch_bounds__(kThreads, 4)
sm70_fd_hmma_reduce1_kernel(
    const float* __restrict__ ws,  // [R, kH, splits, q_rows, kRatio, kD+2]
    float* __restrict__ part,       // [kSplitChunks, R, kH, q_rows, kRatio, kD+2]
    int splits, int q_rows, int R) {
  const int h = blockIdx.x;
  const int r = blockIdx.y;
  const int zs = (kRatio * kD + kThreads - 1) / kThreads;  // 12
  const int ch = blockIdx.z / zs;
  const int slot = (blockIdx.z % zs) * kThreads + threadIdx.x;
  if (slot >= kRatio * kD) {
    return;
  }
  const int i = slot / kD;
  const int d = slot % kD;
  const int64_t psz = (int64_t)q_rows * kRatio * kRowStride;
  const int cs = (splits + kSplitChunks - 1) / kSplitChunks;
  const int s0 = ch * cs;
  const int s1 = min(s0 + cs, splits);
  // ws block for (r, h) starts at (r*kH + h)*splits splits; this CTA
  // covers splits [s0, s1) within it.
  const int64_t w_off =
      (((int64_t)(r * kH + h) * splits) + s0) * psz +
      (int64_t)i * kRowStride;
  const int64_t p_off =
      ((int64_t)ch * R + r) * kH * psz + h * psz + (int64_t)i * kRowStride;

  for (int qq = 0; qq < q_rows; qq++) {
    const float* wq =
        ws + w_off + (int64_t)qq * (kRatio * kRowStride);
    float m_max = -INFINITY;
#pragma unroll 4
    for (int s = s0; s < s1; s++) {
      m_max = fmaxf(m_max, wq[(s - s0) * psz + kD]);
    }
    float l = 0.0f;
    float acc = 0.0f;
    if (m_max != -INFINITY) {
#pragma unroll 4
      for (int s = s0; s < s1; s++) {
        const float* w = wq + (s - s0) * psz;
        const float w_s = expf(w[kD] - m_max);
        l += w_s * w[kD + 1];
        acc += w_s * w[d];
      }
    }
    float* p = part + p_off + (int64_t)qq * (kRatio * kRowStride);
    p[d] = acc;
    p[kD] = m_max;
    p[kD + 1] = l;
  }
}

template <int kH, int kHQ>
__global__ void __launch_bounds__(kThreads, 4)
sm70_fd_hmma_reduce2_kernel(
    __half* __restrict__ out,  // [R, q_rows, kHQ, kD]
    const float* __restrict__ part,
    int splits, int q_rows, int R) {
  const int r = blockIdx.y;
  const int h = blockIdx.x;
  const int slot = blockIdx.z * kThreads + threadIdx.x;
  if (slot >= kRatio * kD) {
    return;
  }
  const int i = slot / kD;
  const int d = slot % kD;
  const int64_t psz = (int64_t)q_rows * kRatio * kRowStride;

  for (int qq = 0; qq < q_rows; qq++) {
    const int64_t pbase =
        h * psz + (int64_t)qq * (kRatio * kRowStride) +
        (int64_t)i * kRowStride;
    // Merge the kSplitChunks (m, l, acc) triples.
    float m = -INFINITY, l = 0.0f, sum = 0.0f;
#pragma unroll
    for (int ch = 0; ch < kSplitChunks; ch++) {
      const float* p =
          part + ((int64_t)ch * R + r) * kH * psz + pbase;
      const float mc = p[kD];
      const float M = fmaxf(m, mc);
      const float w_o = (m == -INFINITY) ? 0.0f : expf(m - M);
      const float w_c = (mc == -INFINITY) ? 0.0f : expf(mc - M);
      sum = sum * w_o + p[d] * w_c;
      l = l * w_o + p[kD + 1] * w_c;
      m = M;
    }
    __half o;
    if (m == -INFINITY) {
      o = __float2half(0.0f);
    } else {
      sum = (l > 0.0f) ? sum / l : 0.0f;
      o = __float2half(sum);
    }
    out[((int64_t)(r * q_rows + qq) * kHQ + h * kRatio + i) * kD + d] = o;
  }
}

}  // namespace fdhmma

using namespace fdhmma;

template <int kH, int kHQ>
void sm70_flashdecode_hmma_impl(torch::stable::Tensor& q,
                      torch::stable::Tensor& kv,
                      torch::stable::Tensor& block_table,
                      torch::stable::Tensor& kv_lens,
                      torch::stable::Tensor& out, double k_scale,
                      double v_scale, double sm_scale, int64_t window_left,
                      int64_t splits, int64_t q_rows,
                      torch::stable::Tensor& workspace) {
  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  // dtypes are asserted by the python wrapper; only shapes are checked here.
  STD_TORCH_CHECK(q.size(1) == q_rows && q_rows >= 1 && q_rows <= kMaxQ,
                  "sm70_flashdecode_hmma: q dim 1 must equal q_rows (1..8)");
  STD_TORCH_CHECK(q.size(2) == kHQ && q.size(3) == kD,
                  "sm70_flashdecode_hmma: q heads/dim mismatch (", kHQ, ")");
  const int64_t H = kv.size(1) / 2;
  const int64_t n_page = kv.size(2);
  STD_TORCH_CHECK(
      H == kH && n_page >= kN && n_page % kN == 0 && kv.size(3) == kF,
      "sm70_flashdecode_hmma: supports H=", kH,
      ", F=144, page = multiple of 16");
  const int64_t R = q.size(0);
  const int64_t bt_cols = block_table.size(1);
  STD_TORCH_CHECK(block_table.size(0) == R && kv_lens.size(0) == R,
                  "sm70_flashdecode_hmma: table/len mismatch");
  STD_TORCH_CHECK(out.size(0) == R && out.size(1) == q_rows &&
                      out.size(2) == kHQ && out.size(3) == kD,
                  "sm70_flashdecode_hmma: out shape mismatch");
  // Workspace layout: [partition ws | reduce stage-1 partials].
  // Partials: [kSplitChunks, R, kH, q_rows, kRatio, kRowStride].
  const int64_t ws_part =
      R * kH * splits * q_rows * kRatio * kRowStride;
  const int64_t ws_red =
      kSplitChunks * R * kH * q_rows * kRatio * kRowStride;
  const int64_t ws_need = ws_part + ws_red;
  STD_TORCH_CHECK(workspace.numel() >= ws_need,
                  "sm70_flashdecode_hmma: workspace too small");
  STD_TORCH_CHECK(splits >= 1, "sm70_flashdecode_hmma: splits < 1");

  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  const int mode = (q_rows >= 2 && q_rows <= kMergedMaxQ)
      ? 2
      : (q_rows >= 5 && q_rows <= kMaxQ ? 3 : 1);
  dim3 grid(splits, kH, R * (mode >= 2 ? 1 : q_rows));
  dim3 block(kRatio * 32);
#define FD_ARGS \
      reinterpret_cast<const __half*>(q.data_ptr()), \
      reinterpret_cast<const uint8_t*>(kv.data_ptr()), \
      reinterpret_cast<const int*>(block_table.data_ptr()), \
      reinterpret_cast<const int*>(kv_lens.data_ptr()), \
      reinterpret_cast<float*>(workspace.data_ptr()), \
      static_cast<float>(k_scale), static_cast<float>(v_scale), \
      static_cast<float>(sm_scale), static_cast<int>(window_left), \
      static_cast<int>(splits), static_cast<int>(q_rows), \
      static_cast<int>(bt_cols), static_cast<int>(n_page), kv.stride(0), \
      (int64_t)H * n_page * kF
  // PV-on-HMMA (T1 lever): run the PV gemm on the 8x8x4 TC pipe (~2x the
  // FFMA pipe on V100). Env-gated; MERGED modes only (host side).
  static const bool kPvHmma = [] {
    const char* e = std::getenv("VLLM_SM70_FD_PV_HMMA");
    return e && e[0] == '1';
  }();
  const bool use_pv = kPvHmma && mode >= 2;
  if (mode == 2) {
    if (use_pv)
      sm70_fd_hmma_partition_kernel_impl<2, kH, kHQ, true><<<grid, block, 0, stream>>>(FD_ARGS);
    else
      sm70_fd_hmma_partition_kernel_impl<2, kH, kHQ, false><<<grid, block, 0, stream>>>(FD_ARGS);
  } else if (mode == 3) {
    if (use_pv)
      sm70_fd_hmma_partition_kernel_impl<3, kH, kHQ, true><<<grid, block, 0, stream>>>(FD_ARGS);
    else
      sm70_fd_hmma_partition_kernel_impl<3, kH, kHQ, false><<<grid, block, 0, stream>>>(FD_ARGS);
  } else {
    sm70_fd_hmma_partition_kernel_impl<1, kH, kHQ, false><<<grid, block, 0, stream>>>(FD_ARGS);
  }
#undef FD_ARGS

  const float* ws_ptr = reinterpret_cast<const float*>(workspace.data_ptr());
  float* part_ptr =
      reinterpret_cast<float*>(workspace.data_ptr()) + ws_part;
  const int zslots = (kRatio * kD + kThreads - 1) / kThreads;

  dim3 rgrid1(kH, R, kSplitChunks * zslots);
  sm70_fd_hmma_reduce1_kernel<kH, kHQ><<<rgrid1, kThreads, 0, stream>>>(
      ws_ptr, part_ptr, static_cast<int>(splits),
      static_cast<int>(q_rows), static_cast<int>(R));

  dim3 rgrid2(kH, R, zslots);
  sm70_fd_hmma_reduce2_kernel<kH, kHQ><<<rgrid2, kThreads, 0, stream>>>(
      reinterpret_cast<__half*>(out.mutable_data_ptr()), part_ptr,
      static_cast<int>(splits), static_cast<int>(q_rows),
      static_cast<int>(R));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "sm70_flashdecode launch failed: ",
                  cudaGetErrorString(err));
}

void sm70_flashdecode_hmma(torch::stable::Tensor& q, torch::stable::Tensor& kv,
                       torch::stable::Tensor& block_table,
                       torch::stable::Tensor& kv_lens,
                       torch::stable::Tensor& out, double k_scale,
                       double v_scale, double sm_scale, int64_t window_left,
                       int64_t splits, int64_t q_rows,
                       torch::stable::Tensor& workspace) {
  // Head geometry is baked into each instantiation: TP2 = (H, HQ) = (2, 12),
  // TP4 = (1, 6); kRatio = 6 in both.
  const int64_t H = kv.size(1) / 2;
  const int64_t HQ = q.size(2);
  if (H == 2 && HQ == 12) {
    sm70_flashdecode_hmma_impl<2, 12>(q, kv, block_table, kv_lens, out,
                                     k_scale, v_scale, sm_scale,
                                     window_left, splits, q_rows, workspace);
  } else if (H == 1 && HQ == 6) {
    sm70_flashdecode_hmma_impl<1, 6>(q, kv, block_table, kv_lens, out,
                                    k_scale, v_scale, sm_scale,
                                    window_left, splits, q_rows, workspace);
  } else {
    STD_TORCH_CHECK(false, "sm70_flashdecode_hmma: supports (H,HQ)=", H,
                    ",", HQ, " (TP2 (2,12) or TP4 (1,6)), D=256");
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_fd_hmma_ops) {
  sm70_fd_hmma_ops.def(
      "sm70_flashdecode_hmma(Tensor q, Tensor kv_cache, Tensor block_table, "
      "Tensor kv_lens, Tensor out, float k_scale, float v_scale, "
      "float sm_scale, int window_left, int splits, int q_rows, "
      "Tensor workspace) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_fd_hmma_ops) {
  sm70_fd_hmma_ops.impl("sm70_flashdecode_hmma",
                            TORCH_BOX(&::sm70_flashdecode_hmma));
}
