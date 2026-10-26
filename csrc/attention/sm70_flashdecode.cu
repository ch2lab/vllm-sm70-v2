// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// SM70 (V100) unified NVFP4 paged decode/verify — FlashDecoding split-KV
// kernel.
//
// Structure (cudagraph-safe, fixed grid):
//   partition kernel  grid = (splits, H, R), block = kRatio*rpb warps
//     (rpb = min(q_rows, 2)); one (kv head, KV split) per block; the block
//     owns the full qo-head group of its kv head, so every KV byte is read
//     exactly once per kv head (no 12x qo-head re-reads, and verify rows
//     SHARE the staged KV: row-pair pass p covers rows [2p, 2p+2) with
//     warp = (qo head, local row), so q=4 pays 1x KV traffic, not 4x).
//     Warp w scores head w with one
//     5-shfl butterfly (lane owns 8 dims) — no cross-warp score merge. Row qq
//     sits at sequence position kv_len - q_rows + qq and attends causally to
//     [lo, kv_len - q_rows + qq + 1).
//     In-kernel NVFP4 dequant (direct 16B vectorized loads + bit ops +
//     LUT — no gather-based indexing, the sm_70 Triton miscompilation that
//     caps the legacy decode at ~1.8 GB/s) and register-prefetched double-
//     buffered 16B page staging (cp.async needs sm_80+; on sm_70 the next
//     chunk's loads are issued into registers before the current chunk's
//     compute, hiding global latency that a synchronous stage serialized).
//     Emits per-(q row, qo head) partial online-softmax states
//     (acc | m | l) to a persistent fp32 workspace.
//   reduce kernel     grid = (H, R, slot-chunks), block = 128 threads
//     log-sum-exp merge across splits into the final fp16 output.
//
// Layout (LBHNC raw bytes, identical to sm70_nvfp4_decode /
// sm70_nvfp4_dequant): one page (stride page) = K side | V side; each side
// = data region (H*N*DD) then scale region (H*N*SD); token row =
// [DD data bytes | SD scale bytes] with F = DD + SD. Nibble: even dim =
// low nibble of byte d//2, odd = high. One e4m3 scale byte per 16 dims.
// Dequant: x = E2M1[nib] * E4M3[sf] * layer_scale.

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

// Head geometry is per-instantiation (template parameters of the kernels and
// host launch): TP2 = (kH, kHQ) = (2, 12); TP4 = (1, 6). kRatio is shared.
constexpr int kRatio = 6;      // qo heads per kv head (kHQ / kH)
constexpr int kD = 256;        // head dim
constexpr int kN = 16;         // tokens per staging chunk (compile-time)
constexpr int kDD = kD / 2;    // 128 data bytes per token row
constexpr int kSD = kD / 16;   // 16 scale bytes per token row
constexpr int kF = kDD + kSD;  // 144 bytes per token row
constexpr int kMaxQ = 8;       // max rows per request
constexpr int kMaxRpb = 2;     // q rows handled per staging pass (shared KV)
constexpr int kThreads = 128;      // reduce-kernel block size
// Partition kernel: kRatio * rpb warps (rpb = min(q_rows, kMaxRpb), passed
// as blockDim); warp = (qo head, local row); each lane owns 8 dims.
constexpr int kPartMaxThreads = kRatio * kMaxRpb * 32;  // 384
constexpr int kMaxPartWarps = kRatio * kMaxRpb;         // 12
constexpr int kLaneDims = 8;
// Workspace slot stride: one (qq, i) row of partial state.
constexpr int kRowStride = kD + 2;

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

// e4m3 scale byte -> float. Matches the Triton LUT semantics (e = 0 is the
// subnormal bin; the store kernel never emits e = 15).
__device__ __forceinline__ float e4m3_to_float(unsigned char b) {
  const bool sgn = b & 0x80;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  float v;
  if (e == 0) {
    v = m * 0.001953125f;  // 2^-9
  } else {
    v = (1.0f + m * 0.125f) * exp2f((float)e - 7.0f);
  }
  return sgn ? -v : v;
}

// e2m1 byte (two nibbles) -> half2 (low nibble = .x). PRMT selects each
// nibble's fp16 magnitude high-byte from an 8-byte table spread over two
// registers (codes 0-3 | codes 4-7); this replaces the smem float2 LUT,
// whose random per-lane bank accesses measured ~35-40% of kernel time.
__device__ __forceinline__ __half2 e2m1_byte_to_h2(uint32_t byte) {
  const uint32_t A = 0x3E3C3800u;  // hi-bytes of 0, .5, 1, 1.5
  const uint32_t B = 0x46444240u;  // hi-bytes of 2, 3, 4, 6
  const uint32_t sel = (byte & 0x7u) | ((byte >> 4) & 0x7u) << 8;
  const uint32_t v = __byte_perm(A, B, sel);
  const uint32_t sgn = ((byte << 12) & 0x00008000u) |
                       ((byte << 24) & 0x80000000u);
  const uint32_t u = ((v << 8) & 0xFF00FF00u) | sgn;
  return *reinterpret_cast<const __half2*>(&u);
}

// Partition kernel: one CTA per (kv head, KV split, request). blockDim =
// kRatio * rpb * 32 (rpb = min(q_rows, kMaxRpb) rows staged per pass), so
// all q rows of a verify batch SHARE one KV staging stream: the warp role
// is (qo head, local row) and every global KV byte is read once per row-
// pair pass instead of once per row (the per-row CTA design paid q_rows x
// traffic). Lane l owns dims [8l, 8l+8); one warp butterfly (5 shfl)
// yields the full 256-dim score per token. Online softmax stays per warp
// (m, l in registers); p is staged through per-warp smem for the PV pass.
// Page staging is double-buffered with register prefetch: the next
// chunk's 16B loads are issued after the current chunk's smem store and
// land before its compute (cp.async is sm_80+; this is the sm_70
// equivalent).
template <int kH, int kHQ>
__global__ void __launch_bounds__(kPartMaxThreads, 2)  // v3g: no-spill budget
sm70_flashdecode_partition_kernel(
    const __half* __restrict__ q,          // [R, q_rows, kHQ, kD]
    const uint8_t* __restrict__ kv,        // raw LBHNC bytes
    const int* __restrict__ bt,            // [R, bt_cols]
    const int* __restrict__ kv_lens,       // [R]
    float* __restrict__ ws,                // [R, kH, splits, q_rows, kRatio, kD+2]
    float k_scale, float v_scale, float sm_scale,
    int window_left, int splits, int q_rows, int bt_cols, int n_page,
    int64_t page_stride, int64_t side_off) {
  const int r = blockIdx.z / q_rows;    // v3h: per-row CTA
  const int qq0 = blockIdx.z % q_rows;  // this CTA's single row
  const int h = blockIdx.y;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int nwarps = blockDim.x >> 5;
  const int rpb = nwarps / kRatio;   // v3h: always 1 (per-row CTA)
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int head_i = warp % kRatio;   // qo head within the kv group
  const int qql = warp / kRatio;      // local row within the pass

  __shared__ __half q_smem[kMaxRpb][kRatio * kD];   // 6 KB (two rows)
  __shared__ uint8_t kv_raw[2][kN * kF];            // v3e: sync staging
  __shared__ float sf_smem[2][kN * kSD];            // v3e
  __shared__ float s_acc_sh[kMaxPartWarps][kN];     // reduced score [w][t]
  __shared__ float p_sh[kMaxPartWarps][kN];         // exp(s - m_new) [w][t]
  __shared__ float stat_sh[2 * kMaxPartWarps];      // w_o, w_p (final merge)
  // v3d: e2m1 pair table back (bisect PRMT out of the suspects).
  __shared__ float2 e2t[256];
  // e4m3 scale table with layer scale folded in: [side*256 + byte] -> float.
  __shared__ float e4t[512];

  // The lane's 8 dims share one 16-dim scale group and four contiguous
  // data bytes in the 128B row.
  const int d_base = kLaneDims * lane;
  const int sg = lane >> 1;

  const int kv_len = __ldg(kv_lens + r);

  // Per-(r,h) workspace block stride (q_rows-dependent: only the active
  // rows are allocated, so q=1 decode needs a small persistent workspace).
  const int64_t ws_sp = (int64_t)q_rows * kRatio * kRowStride;

  const int64_t ws_base =
      ((int64_t)(r * kH + h) * splits + sp) * ws_sp;

  auto write_empty_row = [&](int qq) {
    float* row = ws + ws_base + qq * (kRatio * kRowStride);
    for (int idx = tid; idx < kRatio * kRowStride; idx += blockDim.x) {
      const int rem = idx % kRowStride;
      row[idx] = (rem == kD) ? -INFINITY
               : (rem == kD + 1) ? 0.0f
                                 : (rem < kD ? 0.0f : 0.0f);
    }
  };

  int lo = window_left >= 0 ? kv_len - 1 - window_left : 0;
  if (lo < 0) lo = 0;
  const bool empty_cta =
      kv_len == 0 || lo >= kv_len;
  const int64_t t_a =
      (int64_t)lo + ((int64_t)kv_len - lo) * sp / splits;
  const int64_t t_b =
      (int64_t)lo + ((int64_t)kv_len - lo) * (sp + 1) / splits;

  // Each warp owns rows {qql, qql+rpb, ...}: init its own slots once,
  // before any merge reads them. No cross-warp slot sharing.
  write_empty_row(qq0);
  if (empty_cta || t_a >= t_b) return;

  const int c0 = (int)(t_a / kN);
  const int c1 = (int)((t_b - 1) / kN);

  // Resolve one 16B chunk index (c in [0, 288)) of chunk grid slot cbase
  // (tokens [cbase, cbase+16)) to its global source pointer.
  constexpr int kChunks = 2 * kN * (kF / 16);  // 288
  auto chunk_src = [&](int c, int n0, const int64_t base) -> const uint4* {
    const int rem = c % (kN * kF / 16);
    const int s = c / (kN * kF / 16);
    const int n = rem / (kF / 16);
    const int f16 = rem % (kF / 16);
    int64_t off;
    if (f16 < kDD / 16) {  // data chunk
      off = (int64_t)h * (n_page * kDD) + (n0 + n) * kDD + f16 * 16;
    } else {  // scale chunk (exactly one 16B chunk per token row)
      off = (int64_t)kH * (n_page * kDD) + (int64_t)h * (n_page * kSD) +
            (n0 + n) * kSD + (f16 - kDD / 16) * 16;
    }
    return reinterpret_cast<const uint4*>(kv + base + s * side_off + off);
  };
  auto chunk_dst = [&](int c) -> uint4* {
    const int rem = c % (kN * kF / 16);
    const int s = c / (kN * kF / 16);
    const int n = rem / (kF / 16);
    const int f16 = rem % (kF / 16);
    return reinterpret_cast<uint4*>(&kv_raw[s][n * kF + f16 * 16]);
  };

  // Stage one 16-token chunk: load into regs then store (v3e: WIP-sync).
  auto stage_chunk = [&](int c) {
    const int cp = c * kN;
    const int b = cp / n_page;
    const int n0 = cp - b * n_page;
    const int64_t base =
        (int64_t)__ldg(bt + r * bt_cols + b) * page_stride;
    if (tid < kChunks) *chunk_dst(tid) = *chunk_src(tid, n0, base);
    if (tid + blockDim.x < kChunks) {
      *chunk_dst(tid + blockDim.x) = *chunk_src(tid + blockDim.x, n0, base);
    }
    __syncthreads();
  };

  // Decode all scale bytes of the staged chunk (layer scale pre-multiplied).
  auto decode_scales = [&]() {
    constexpr int kScales = 2 * kN * kSD;  // 512
    for (int idx = tid; idx < kScales; idx += blockDim.x) {
      const int s = idx / (kN * kSD);
      const int rem = idx % (kN * kSD);
      const int n = rem / kSD;
      const int g = rem % kSD;
      sf_smem[s][n * kSD + g] =
          e4t[s * 256 + kv_raw[s][n * kF + kDD + g]];
    }
    __syncthreads();
  };

  // Load the q rows of this pass into q_smem (sm_scale folded in).
  auto load_q_rows = [&](int row_base) {
    const int nrows = min(rpb, q_rows - row_base);
    for (int qq = 0; qq < nrows; qq++) {
      const __half* src =
          q + ((int64_t)(r * q_rows + row_base + qq) * kHQ + h * kRatio) * kD;
      for (int idx = tid; idx < kRatio * kD / 2; idx += blockDim.x) {
        const int t = idx / (kD / 2);
        const int d2 = (idx % (kD / 2)) * 2;
        const __half2 v = *reinterpret_cast<const __half2*>(src + t * kD + d2);
        const __half2 s = __float2half2_rn(sm_scale);
        reinterpret_cast<__half2*>(&q_smem[qq][0])[idx] = __hmul2(v, s);
      }
    }
    __syncthreads();
  };

  // Build the dequant LUTs once (v3d: e2t restored).
  for (int b = tid; b < 256; b += blockDim.x) {
    e2t[b] = make_float2(e2m1_f(b & 0xF), e2m1_f(b >> 4));
    e4t[b] = e4m3_to_float(b) * k_scale;
    e4t[256 + b] = e4m3_to_float(b) * v_scale;
  }
  __syncthreads();

  for (int pass = 0; pass < 1; pass++) {
    const int row_base = 0;
    const int qq = qq0;            // the CTA's row; qql must be 0 to act
    const bool active = qql == 0;

    // Online-softmax state for this (head, row).
    float acc[kLaneDims];
#pragma unroll
    for (int j = 0; j < kLaneDims; j++) acc[j] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    const int hi = kv_len - q_rows + qq0 + 1;

    load_q_rows(qq0);  // q_smem[0] holds THIS CTA's row

    // Hoist this warp's q slice (head head_i, local row) to registers.
    float qf[kLaneDims];
    if (active) {
      const __half* qh = &q_smem[0][head_i * kD + d_base];
#pragma unroll
      for (int j = 0; j < kLaneDims; j++) qf[j] = __half2float(qh[j]);
    }
    __syncthreads();  // q_smem reuse across passes

    // v3e: synchronous chunk sweep (WIP-style).
    for (int c = c0; c <= c1; c++) {
      stage_chunk(c);
      decode_scales();

      const int cbase = c * kN;
      const int t0 = cbase > (int)t_a ? cbase : (int)t_a;
      const int t1_page = cbase + kN < (int)t_b ? cbase + kN : (int)t_b;
      const int t1 = t1_page < hi ? t1_page : hi;

      if (active && t0 < t1) {
        // Phase 1: per-token 256-dim score via one warp butterfly.
        const int w = warp;
        float m_tile = -INFINITY;
        for (int t = t0; t < t1; t++) {
          const int n = t - cbase;
          const uint32_t kb4 = *reinterpret_cast<const uint32_t*>(
              kv_raw[0] + n * kF + 4 * lane);
          const float sf = sf_smem[0][n * kSD + sg];
          const float2 kl[4] = {e2t[kb4 & 0xFF], e2t[(kb4 >> 8) & 0xFF],
                                e2t[(kb4 >> 16) & 0xFF], e2t[kb4 >> 24]};
          float s = 0.0f;
#pragma unroll
          for (int b = 0; b < 4; b++) {
            s = fmaf(qf[2 * b], kl[b].x, s);
            s = fmaf(qf[2 * b + 1], kl[b].y, s);
          }
          s *= sf;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(0xFFFFFFFF, s, o);
          }
          if (lane == 0) s_acc_sh[w][t - t0] = s;
          m_tile = fmaxf(m_tile, s);
        }
        __syncwarp();
        // v3d: serial lane-0 softmax update (WIP semantics).
        float alpha_b = 1.0f;
        if (lane == 0) {
          const float m_new = fmaxf(m, m_tile);
          alpha_b = (m == -INFINITY) ? 0.0f : expf(m - m_new);
          l *= alpha_b;
          for (int t = t0; t < t1; t++) {
            const float p = expf(s_acc_sh[w][t - t0] - m_new);
            p_sh[w][t - t0] = p;
            l += p;
          }
          m = m_new;
        }
        __syncwarp();
        alpha_b = __shfl_sync(0xFFFFFFFF, alpha_b, 0);
#pragma unroll
        for (int j = 0; j < kLaneDims; j++) acc[j] *= alpha_b;
        // Phase 2: PV, 8 dims per lane.
        for (int t = t0; t < t1; t++) {
          const int n = t - cbase;
          const uint32_t vb4 = *reinterpret_cast<const uint32_t*>(
              kv_raw[1] + n * kF + 4 * lane);
          const float2 vl[4] = {e2t[vb4 & 0xFF], e2t[(vb4 >> 8) & 0xFF],
                                e2t[(vb4 >> 16) & 0xFF], e2t[vb4 >> 24]};
          const float ps = p_sh[w][t - t0] * sf_smem[1][n * kSD + sg];
#pragma unroll
          for (int b = 0; b < 4; b++) {
            acc[2 * b] = fmaf(ps, vl[b].x, acc[2 * b]);
            acc[2 * b + 1] = fmaf(ps, vl[b].y, acc[2 * b + 1]);
          }
        }
      }
    }

    // Final LSE merge of this CTA's partial into its (row qq, head head_i)
    // ws slot (all warps take both barriers; inactive warps idle through).
    float* row = ws + ws_base + qq * (kRatio * kRowStride);
    if (active && lane == 0) {
      const float m_ws = row[head_i * kRowStride + kD];
      const float l_ws = row[head_i * kRowStride + kD + 1];
      const float M = fmaxf(m_ws, m);
      float w_o, w_p, l_m;
      if (M == -INFINITY) {
        w_o = 0.0f; w_p = 0.0f; l_m = 0.0f;
      } else {
        w_o = (m_ws == -INFINITY) ? 0.0f : expf(m_ws - M);
        w_p = (m == -INFINITY) ? 0.0f : expf(m - M);
        l_m = w_o * l_ws + w_p * l;
      }
      stat_sh[2 * warp] = w_o;
      stat_sh[2 * warp + 1] = w_p;
      row[head_i * kRowStride + kD] = M;
      row[head_i * kRowStride + kD + 1] = l_m;
    }
    __syncthreads();
    if (active) {
      const float w_o = stat_sh[2 * warp];
      const float w_p = stat_sh[2 * warp + 1];
      float* rp = row + head_i * kRowStride + d_base;
#pragma unroll
      for (int j = 0; j < kLaneDims; j++) {
        rp[j] = w_o * rp[j] + w_p * acc[j];
      }
    }
    __syncthreads();  // stat_sh / q_smem reuse across passes
  }
}

// Two-stage LSE reduce: stage 1 splits the O(splits) read loop across
// kSplitChunks more CTAs (24 CTAs at R=1 latency-bound the whole reduce at
// ~40GB/s effective; 4x CTAs -> ~4x), emitting per-chunk (m, l, acc)
// triples; stage 2 merges the kSplitChunks triples (4 iterations) into out.
constexpr int kSplitChunks = 4;

template <int kH, int kHQ>
__global__ void __launch_bounds__(kThreads, 4)
sm70_flashdecode_reduce1_kernel(
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
sm70_flashdecode_reduce2_kernel(
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

}  // namespace

template <int kH, int kHQ>
void sm70_flashdecode_impl(torch::stable::Tensor& q,
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
                  "sm70_flashdecode: q dim 1 must equal q_rows (1..8)");
  STD_TORCH_CHECK(q.size(2) == kHQ && q.size(3) == kD,
                  "sm70_flashdecode: q heads/dim mismatch (", kHQ, ")");
  const int64_t H = kv.size(1) / 2;
  const int64_t n_page = kv.size(2);
  STD_TORCH_CHECK(
      H == kH && n_page >= kN && n_page % kN == 0 && kv.size(3) == kF,
      "sm70_flashdecode: supports H=", kH,
      ", F=144, page = multiple of 16");
  const int64_t R = q.size(0);
  const int64_t bt_cols = block_table.size(1);
  STD_TORCH_CHECK(block_table.size(0) == R && kv_lens.size(0) == R,
                  "sm70_flashdecode: table/len mismatch");
  STD_TORCH_CHECK(out.size(0) == R && out.size(1) == q_rows &&
                      out.size(2) == kHQ && out.size(3) == kD,
                  "sm70_flashdecode: out shape mismatch");
  // Workspace layout: [partition ws | reduce stage-1 partials].
  // Partials: [kSplitChunks, R, kH, q_rows, kRatio, kRowStride].
  const int64_t ws_part =
      R * kH * splits * q_rows * kRatio * kRowStride;
  const int64_t ws_red =
      kSplitChunks * R * kH * q_rows * kRatio * kRowStride;
  const int64_t ws_need = ws_part + ws_red;
  STD_TORCH_CHECK(workspace.numel() >= ws_need,
                  "sm70_flashdecode: workspace too small");
  STD_TORCH_CHECK(splits >= 1, "sm70_flashdecode: splits < 1");

  cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  dim3 grid(splits, kH, R * q_rows);  // v3h: per-row CTA grid
  dim3 block(kRatio * 32);
  sm70_flashdecode_partition_kernel<kH, kHQ><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(kv.data_ptr()),
      reinterpret_cast<const int*>(block_table.data_ptr()),
      reinterpret_cast<const int*>(kv_lens.data_ptr()),
      reinterpret_cast<float*>(workspace.data_ptr()),
      static_cast<float>(k_scale), static_cast<float>(v_scale),
      static_cast<float>(sm_scale), static_cast<int>(window_left),
      static_cast<int>(splits), static_cast<int>(q_rows),
      static_cast<int>(bt_cols), static_cast<int>(n_page), kv.stride(0),
      (int64_t)H * n_page * kF);

  const float* ws_ptr = reinterpret_cast<const float*>(workspace.data_ptr());
  float* part_ptr =
      reinterpret_cast<float*>(workspace.data_ptr()) + ws_part;
  const int zslots = (kRatio * kD + kThreads - 1) / kThreads;

  dim3 rgrid1(kH, R, kSplitChunks * zslots);
  sm70_flashdecode_reduce1_kernel<kH, kHQ><<<rgrid1, kThreads, 0, stream>>>(
      ws_ptr, part_ptr, static_cast<int>(splits),
      static_cast<int>(q_rows), static_cast<int>(R));

  dim3 rgrid2(kH, R, zslots);
  sm70_flashdecode_reduce2_kernel<kH, kHQ><<<rgrid2, kThreads, 0, stream>>>(
      reinterpret_cast<__half*>(out.mutable_data_ptr()), part_ptr,
      static_cast<int>(splits), static_cast<int>(q_rows),
      static_cast<int>(R));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "sm70_flashdecode launch failed: ",
                  cudaGetErrorString(err));
}

void sm70_flashdecode(torch::stable::Tensor& q, torch::stable::Tensor& kv,
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
    sm70_flashdecode_impl<2, 12>(q, kv, block_table, kv_lens, out, k_scale,
                                 v_scale, sm_scale, window_left, splits,
                                 q_rows, workspace);
  } else if (H == 1 && HQ == 6) {
    sm70_flashdecode_impl<1, 6>(q, kv, block_table, kv_lens, out, k_scale,
                                v_scale, sm_scale, window_left, splits,
                                q_rows, workspace);
  } else {
    STD_TORCH_CHECK(false, "sm70_flashdecode: supports (H,HQ)=", H, ",",
                    HQ, " (TP2 (2,12) or TP4 (1,6)), D=256");
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_flashdecode_ops) {
  sm70_flashdecode_ops.def(
      "sm70_flashdecode(Tensor q, Tensor kv_cache, Tensor block_table, "
      "Tensor kv_lens, Tensor out, float k_scale, float v_scale, "
      "float sm_scale, int window_left, int splits, int q_rows, "
      "Tensor workspace) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_flashdecode_ops) {
  sm70_flashdecode_ops.impl("sm70_flashdecode",
                            TORCH_BOX(&::sm70_flashdecode));
}
