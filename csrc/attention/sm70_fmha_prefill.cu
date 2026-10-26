// T12: custom sm70 TC dense prefill attention (WMMA m16n16k16) for the 12
// full-attention layers of Qwen3.8-Flash-Next at TP8 (3 q heads / 1 kv head
// per rank, D=256, fp16 paged KV, chunked 65K prefill).
//
// Why: the serving mem-eff kernel (fmha_cutlassF 32x128 rf sm70) runs at
// 2-21% of the V100 fp16-TC peak on the current chunked workload; this
// kernel targets >=2x via online-softmax flash structure + split-KV
// (per-CTA kv partition + LSE partial merge).
//
// Standalone JIT extension (torch::Tensor + PYBIND11_MODULE, same pattern as
// csrc/libtorch_stable/qsa/qsa_prefill_sm70.cu): offline bench/parity gate
// first; in-service wiring is a later, separate step.
//
// b-1 (2026-10-18) geometry work, offline:
//  * C2 (QROWS 16 -> 32 = 96 M-rows, 12 warps, half the CTA count and hence
//    half the K/V global re-read) measured 0.92x at kv=16384/65536. The
//    traffic did halve, so the kernel is NOT HBM-bandwidth-bound: it is
//    latency/occupancy-bound. v1's 2 CTA/SM is what hides the staging.
//  * PIPE (this revision) attacks the exposed load instead of the traffic:
//    the chunk's LDGs are issue-early (chunk c+1 loaded into registers during
//    chunk c's QK/softmax/PV) so the ~HBM latency overlaps compute. Register
//    staging only - no extra shared memory - so the 2 CTA/SM occupancy is
//    preserved (V100 has no cp.async, so this is the only prefetch form).
//
// Contract (per launch, R = batch, Q = q_rows tokens per request):
//   q:          fp16 [R, Q, 3, 256] contiguous
//   kv_cache:   fp16 [B, 2, N, 1, 256] (K side 0, V side 1)
//   block_table: int32 [R, max_blocks]
//   kv_lens:    int32 [R]  (visible kv per request)
//   out:        fp16 [R, Q, 3, 256]   (written in place)
//   ws:         fp32 [R, Q, 3, S, 258]  (per-(row, head, split) partial:
//                                        O fp32 + m + l)
// Causal: row r (0-based within the request) attends to kv[0, hi) with
//   hi = kv_len - Q + r + 1   (the chunk's rows sit at the TAIL of the
//   visible kv, exactly the sm70_flashdecode hi_row semantics).
//
// Geometry (v1, TP8-fixed):
//   CTA = 16 tokens x 3 heads = 48 M-rows (3 wmma m-tiles of 16), 6 warps
//   (warp 0..2 compute scores for m-tile warp; warp = (m-tile, d-half 128)
//   for PV). kv chunk = 16 tokens; grid = (splits, 1, R * (Q/16)).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

using namespace nvcuda;  // wmma resolves in kernel bodies (sm70, nvcc 12.8)

namespace {

constexpr int kD = 256;
constexpr int kHQ = 3;
constexpr int kN = 16;        // kv tokens per chunk
constexpr int kDV = 128;      // d-half per PV warp
// smem row stride for the [rows][kD] K/V/Q panels: 256 halfs (512B) lands
// every row start on the same 16B bank group (16-way wmma load conflicts);
// 264 (528B = 33 bank units, coprime with 32) deconflicts the aQ/aK/aV
// operand loads. +8 halfs/row = +1.3KB total.
constexpr int kDS = kD + 8;
constexpr int kSub = kD / 64;  // 32 half2 sub-iterations (4 rows) per panel

// 16x16 fp32 accumulator fragment -> (row, col). Probed on this toolchain
// (nvcc 12.8, sm_70) with a single-mma layout probe and verified against a
// host matmul on 5 random trials (sm70_tests/scratch_wmma_single_probe.cu):
// each lane owns rows {b, b+2} x 4 col pairs; the textbook row/col bit
// selectors differ from what this toolchain emits.
__device__ __forceinline__ int frag_row(int lane, int j) {
  return 4 * (((lane >> 4) & 1) + ((lane & 4) >> 1)) + (lane & 1) +
         2 * ((j >> 1) & 1);
}
__device__ __forceinline__ int frag_col(int lane, int j) {
  return (lane & 8) + (lane & 2) + 4 * ((j >> 2) & 1) + (j & 1);
}

// ---- panel staging, split so it can be software-pipelined ----
// Row-serial mapping: a warp instruction covers 1 row x 32 half2 -> 1-way smem
// stores (32 distinct 4B banks). Each warp loads ALL of its rows (<=3 x 4 LDG)
// into static-indexed register buffers (bA/bB/bC) BEFORE any store: per-row
// global bases are hoisted (one warp-uniform block_table lookup per row, <=3
// per panel). The register rotation between rows is all static-indexed copies
// -> no local memory. (v3.2's rolling buf[k & 7] had a dynamic index, ptxas
// spilled it to local (STACK:128) and regressed 2.15x; v3.1's per-row 4-LDG
// (16B/lane) kept staging at 41% of wall.)
template <int kW>
__device__ __forceinline__ void ldg_kv_panel(
    const __half* __restrict__ kv_cache, const int* __restrict__ block_table,
    int r, int bt_cols, int n_page, int64_t kv_s0, int cbase, int warp, int lane,
    bool v_side, __half2 (&bA)[kSub], __half2 (&bB)[kSub], __half2 (&bC)[kSub]) {
  const int nrow = (kN - 1 - warp) / kW + 1;  // rows warp, warp+kW, ...
  const int tok0 = cbase + warp;
  const int tok1 = cbase + warp + kW;
  const int tok2 = cbase + warp + 2 * kW;
  const int64_t vside = v_side ? (int64_t)n_page * kD : 0;
  const __half* rb0 =
      kv_cache + (int64_t)__ldg(block_table + r * bt_cols + tok0 / n_page) *
                     kv_s0 +
      vside + (tok0 - (tok0 / n_page) * n_page) * kD;
  const __half* rb1 = (nrow > 1)
      ? kv_cache + (int64_t)__ldg(block_table + r * bt_cols + tok1 / n_page) *
                       kv_s0 +
        vside + (tok1 - (tok1 / n_page) * n_page) * kD
      : rb0;
  const __half* rb2 = (nrow > 2)
      ? kv_cache + (int64_t)__ldg(block_table + r * bt_cols + tok2 / n_page) *
                       kv_s0 +
        vside + (tok2 - (tok2 / n_page) * n_page) * kD
      : rb0;
#pragma unroll
  for (int s = 0; s < kSub; s++)
    bA[s] = *reinterpret_cast<const __half2*>(rb0 + s * 64 + lane * 2);
  if (nrow > 1) {
#pragma unroll
    for (int s = 0; s < kSub; s++)
      bB[s] = *reinterpret_cast<const __half2*>(rb1 + s * 64 + lane * 2);
  }
  if (nrow > 2) {
#pragma unroll
    for (int s = 0; s < kSub; s++)
      bC[s] = *reinterpret_cast<const __half2*>(rb2 + s * 64 + lane * 2);
  }
}

template <int kW>
__device__ __forceinline__ void sts_kv_panel(__half* sPanel, int warp, int lane,
                                             __half2 (&bA)[kSub],
                                             __half2 (&bB)[kSub],
                                             __half2 (&bC)[kSub]) {
  const int nrow = (kN - 1 - warp) / kW + 1;
  for (int ri = 0; ri < nrow; ri++) {
    const int rr = warp + ri * kW;
#pragma unroll
    for (int s = 0; s < kSub; s++)
      *reinterpret_cast<__half2*>(&sPanel[rr * kDS + s * 64 + lane * 2]) = bA[s];
    if (ri + 1 < nrow) {
#pragma unroll
      for (int s = 0; s < kSub; s++)
        bA[s] = bB[s];
      if (ri + 2 < nrow) {
#pragma unroll
        for (int s = 0; s < kSub; s++)
          bB[s] = bC[s];
      }
    }
  }
}

// QROWS = tokens per CTA; MINB = min blocks/SM for __launch_bounds__;
// PIPE = 1 -> issue chunk c+1's LDGs during chunk c's compute.
// ABL = ablation ladder for offline phase attribution (diagnosis only, never
// wired to serving): 0 = full, 1 = drop PV, 2 = drop softmax+PV, 3 = stage only.
// A dead phase would let ptxas eliminate the staging it feeds, so the live
// consumers are re-established by one unreachable store after the loop
// (outside the timed region) - see the ABL>=1 keep-alive below.
template <int QROWS, int MINB, int PIPE, int ABL>
__global__ void __launch_bounds__(QROWS * kHQ * 4, MINB)
fmha_prefill_wmma_kernel(const __half* __restrict__ q,
                         const __half* __restrict__ kv_cache,
                         const int* __restrict__ block_table,
                         const int* __restrict__ kv_lens,
                         float* __restrict__ ws,
                         float sm_scale,
                         int splits, int q_rows, int n_page, int bt_cols,
                         int64_t kv_s0) {
  constexpr int kMR = QROWS * kHQ;   // M-rows per CTA (48 / 96)
  constexpr int kMT = kMR / 16;      // m-tiles (3 / 6)
  constexpr int kW = kMT * 2;        // warps (6 / 12)
  constexpr int kT = kW * 32;        // threads (192 / 384)
  // All panels are carved from ONE dynamic allocation: sm_70 static __shared__
  // is capped at 48 KB, and the 32-token geometry needs ~78 KB
  // (cudaFuncAttributeMaxDynamicSharedMemorySize opt-in in the launcher).
  // (nvcuda::wmma_init() is a no-op on this toolchain; the QSA wmma probes
  // on this box run correctly without it.)
  extern __shared__ __align__(16) unsigned char smem[];
  __half* sQ = reinterpret_cast<__half*>(smem);
  __half* sK = sQ + kMR * kDS;
  __half* sV = sK + kN * kDS;
  // sScore row stride 20 (80B): the HW store_matrix_sync row writes are
  // 16B-vectorized, so ldm*4 must be 16B-aligned (17 = 68B faults with
  // misaligned address; 20 verified by the aprobe e2e gate).
  float* sScore = reinterpret_cast<float*>(sV + kN * kDS);
  __half* sP = reinterpret_cast<__half*>(sScore + kMR * (kN + 4));
  float* sRowMax = reinterpret_cast<float*>(sP + kMR * kN);
  float* sRowSum = sRowMax + kMR;
  float* sAlpha = sRowSum + kMR;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int qtile = blockIdx.z % (q_rows / QROWS);
  const int r = blockIdx.z / (q_rows / QROWS);
  const int sp = blockIdx.x;
  const int row0 = qtile * QROWS;
  const int kv_len = __ldg(kv_lens + r);

  // ---- split partition of the chunk index range [0, n_chunks) ----
  // Partition on chunk indices (not tokens) so no 16-token chunk is ever
  // covered by two splits (token-proportional bounds overlap when
  // kv_len/splits is not a multiple of kN).
  const int64_t n_chunks = (kv_len + kN - 1) / kN;
  const int c0 = (int)(n_chunks * sp / splits);
  const int c1 = (int)(n_chunks * (sp + 1) / splits);  // exclusive

  // ---- Q panel: M-row i = token (row0 + i/kHQ), head (i%kHQ) ----
  for (int i = tid; i < kMR * kD / 2; i += kT) {
    const int m = i / (kD / 2), d2 = (i % (kD / 2)) * 2;
    const int tok = row0 + m / kHQ;
    const int hq = m % kHQ;
    *reinterpret_cast<__half2*>(&sQ[m * kDS + d2]) =
        *reinterpret_cast<const __half2*>(
            q + (((int64_t)(r * q_rows + tok) * kHQ + hq) * kD) + d2);
  }
  for (int i = tid; i < kMR; i += kT) {
    sRowMax[i] = -INFINITY;
    sRowSum[i] = 0.0f;
    sAlpha[i] = 1.0f;
  }
  __syncthreads();

  // ws partial slot for (r, row, head, sp): ws [R, Q, 3, S, 258]; every
  // element is written exactly once by the end-of-kernel partial write.
  // ws [R, Q, 3(hq), S, kD+2]: per (r, qrow) the region is [hq][split][kD+2];
  // M-row m = (tok = m/kHQ, hq = m%kHQ) maps to qrow row0+tok, head hq.
  const int64_t wsz_row = kHQ * splits * (kD + 2);
  const int64_t hq_stride = splits * (kD + 2);
  const int64_t sp_off = sp * (kD + 2);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aQ;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> aK;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aP;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> aV;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kDV / 16];

  // Every warp owns (mt, dh) = (warp%kMT, warp/kMT) for PV, so all kW warps
  // cover the kMT m-tiles x 2 d-halves of the 256-dim output.
  const int dh = warp / kMT;
  const int mt = warp % kMT;
  for (int dt = 0; dt < kDV / 16; dt++)
    wmma::fill_fragment(acc[dt], 0.0f);

  // Register staging buffers (both K and V; used as prefetch targets in PIPE).
  __half2 kA[kSub], kB[kSub], kC[kSub];
  __half2 vA[kSub], vB[kSub], vC[kSub];

  if (PIPE && c1 > c0) {
    // Prologue: fill the registers for the first chunk of this CTA's range.
    ldg_kv_panel<kW>(kv_cache, block_table, r, bt_cols, n_page, kv_s0, c0 * kN,
                     warp, lane, false, kA, kB, kC);
    ldg_kv_panel<kW>(kv_cache, block_table, r, bt_cols, n_page, kv_s0, c0 * kN,
                     warp, lane, true, vA, vB, vC);
  }

  for (int c = c0; c < c1; c++) {
    const int cbase = c * kN;
    if (PIPE) {
      sts_kv_panel<kW>(sK, warp, lane, kA, kB, kC);
      sts_kv_panel<kW>(sV, warp, lane, vA, vB, vC);
      // Issue the NEXT chunk's loads now: their HBM latency overlaps the
      // QK/softmax/PV below (the STS above already consumed the registers, so
      // the compiler keeps this order; nothing here waits on smem).
      if (c + 1 < c1) {
        ldg_kv_panel<kW>(kv_cache, block_table, r, bt_cols, n_page, kv_s0,
                         (c + 1) * kN, warp, lane, false, kA, kB, kC);
        ldg_kv_panel<kW>(kv_cache, block_table, r, bt_cols, n_page, kv_s0,
                         (c + 1) * kN, warp, lane, true, vA, vB, vC);
      }
    } else {
      ldg_kv_panel<kW>(kv_cache, block_table, r, bt_cols, n_page, kv_s0, cbase,
                       warp, lane, false, kA, kB, kC);
      sts_kv_panel<kW>(sK, warp, lane, kA, kB, kC);
      ldg_kv_panel<kW>(kv_cache, block_table, r, bt_cols, n_page, kv_s0, cbase,
                       warp, lane, true, vA, vB, vC);
      sts_kv_panel<kW>(sV, warp, lane, vA, vB, vC);
    }
    __syncthreads();

    // ---- QK scores + store_matrix_sync publish (dh=0 warps only, v3.1) ----
    // The dh=1 twin would compute an identical score tile: it skips (v2
    // dedup). sc is published through the HW-defined store_matrix_sync
    // layout, which is context-independent: the probed .x (row, col)
    // mapping is a ptxas artifact that changed between v2 and v3 and
    // silently broke the causal mask (v3 G1 fail, m=-inf on every row).
    if (ABL <= 2 && dh == 0) {
      const int sw = warp % kMT;
      wmma::fill_fragment(sc, 0.0f);
#pragma unroll
      for (int kc = 0; kc < kD / 16; kc++) {
        wmma::load_matrix_sync(aQ, &sQ[(sw * 16) * kDS + kc * 16], kDS);
        wmma::load_matrix_sync(aK, &sK[kc * 16], kDS);
        wmma::mma_sync(sc, aQ, aK, sc);
      }
      wmma::store_matrix_sync(&sScore[sw * 16 * (kN + 4)], sc, kN + 4,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // ---- online softmax: 4 threads per M-row, all threads (v3.1) ----
    // v1: 48-thread (1.5-warp) serial phase = 41% of wall (phase
    // decomposition). Now every warp works: quad = 4 consecutive tids of
    // one row (shfl_xor 1, 2), 4 cols per thread.
    if (ABL <= 1) {
      const int m = tid >> 2, cb = (tid & 3) * 4;
      int rt1 = kv_len - q_rows + row0 + m / kHQ + 1 - cbase;
      if (rt1 > kN) rt1 = kN;
      if (rt1 < 0) rt1 = 0;
      const float* sp_ = &sScore[m * (kN + 4) + cb];
      float s0 = sp_[0] * sm_scale;
      float s1 = sp_[1] * sm_scale;
      float s2 = sp_[2] * sm_scale;
      float s3 = sp_[3] * sm_scale;
      if (cb + 0 >= rt1) s0 = -INFINITY;
      if (cb + 1 >= rt1) s1 = -INFINITY;
      if (cb + 2 >= rt1) s2 = -INFINITY;
      if (cb + 3 >= rt1) s3 = -INFINITY;
      float mx = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_old = sRowMax[m];
      const float m_new = fmaxf(m_old, mx);
      const float alpha =
          (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
      const float p0 = (m_new == -INFINITY) ? 0.0f : expf(s0 - m_new);
      const float p1 = (m_new == -INFINITY) ? 0.0f : expf(s1 - m_new);
      const float p2 = (m_new == -INFINITY) ? 0.0f : expf(s2 - m_new);
      const float p3 = (m_new == -INFINITY) ? 0.0f : expf(s3 - m_new);
      float l = p0 + p1 + p2 + p3;
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      if ((tid & 3) == 0) {  // quad leader
        sAlpha[m] = alpha;
        sRowMax[m] = m_new;
        sRowSum[m] = sRowSum[m] * alpha + l;
      }
      sP[m * kN + cb + 0] = __float2half(p0);
      sP[m * kN + cb + 1] = __float2half(p1);
      sP[m * kN + cb + 2] = __float2half(p2);
      sP[m * kN + cb + 3] = __float2half(p3);
    }
    __syncthreads();

    // ---- PV: warp = (mt, dh); O[16m x 128d] += P[16m x 16kv] @ V ----
    if (ABL == 0) {
      // rescale O by the per-row alpha of this chunk. frag_row(lane, j) =
      // fb + 2*((j>>1)&1) with fb = frag_row(lane, 0): each thread touches
      // only TWO alpha values; two smem reads + a compile-time select.
      // (A local float[16] array indexed by runtime frag_row spilled to
      // local memory: NCU est. 58% speedup to remove.)
      const int fb = frag_row(lane, 0);
      const float alpha0 = sAlpha[mt * 16 + fb];
      const float alpha2 = sAlpha[mt * 16 + fb + 2];
#pragma unroll
      for (int dt = 0; dt < kDV / 16; dt++) {
        for (int j = 0; j < (int)acc[dt].num_elements; j++)
          acc[dt].x[j] *= ((j >> 1) & 1) ? alpha2 : alpha0;
      }
      wmma::load_matrix_sync(aP, &sP[mt * 16 * kN], kN);
#pragma unroll
      for (int dt = 0; dt < kDV / 16; dt++) {
        wmma::load_matrix_sync(aV, &sV[dh * kDV + dt * 16], kDS);
        wmma::mma_sync(acc[dt], aP, aV, acc[dt]);
      }
    }
    __syncthreads();
  }

  if (ABL >= 1) {
    // Ablation builds only. With a phase switched off its inputs have no live
    // consumer, so ptxas would eliminate the staging that feeds it and the
    // timing would measure nothing. Re-establish the consumers here, once per
    // CTA, OUTSIDE the timed chunk loop, via a store the compiler cannot prove
    // unreachable.
    float chk = 0.0f;
    for (int i = tid; i < kN * kDS; i += kT) chk += __half2float(sV[i]);
    if (ABL >= 2)
      for (int i = tid; i < kMR * (kN + 4); i += kT) chk += sScore[i];
    if (ABL >= 3)
      for (int i = tid; i < kN * kDS; i += kT) chk += __half2float(sK[i]);
    if (chk == 1.0f / 3.0f) ws[0] = chk;
  }

  // ---- partial write: O frag -> ws; m/l per M-row ----
  // M-row m = (tok = m/kHQ, hq = m%kHQ) -> qrow row0+tok, head hq.
#pragma unroll
  for (int dt = 0; dt < kDV / 16; dt++) {
    for (int j = 0; j < (int)acc[dt].num_elements; j++) {
      const int m = mt * 16 + frag_row(lane, j);
      const int tok = m / kHQ, hq = m % kHQ;
      const int d = dh * kDV + dt * 16 + frag_col(lane, j);
      ws[(((int64_t)(r * q_rows + row0 + tok)) * wsz_row
          + hq * hq_stride + sp_off + d)] = acc[dt].x[j];
    }
  }
  __syncthreads();
  if (tid < kMR) {
    const int tok = tid / kHQ, hq = tid % kHQ;
    float* bm = ws + (((int64_t)(r * q_rows + row0 + tok)) * wsz_row
                      + hq * hq_stride + sp_off);
    bm[kD] = sRowMax[tid];
    bm[kD + 1] = sRowSum[tid];
  }
}

// Merge: one CTA per global row (all 3 heads), 192 threads.
__global__ void __launch_bounds__(192, 8)
fmha_prefill_merge_kernel(const float* __restrict__ ws,
                          __half* __restrict__ out, int splits, int q_rows) {
  const int row = blockIdx.x % q_rows;
  const int r = blockIdx.x / q_rows;
  const int tid = threadIdx.x;
  const int64_t wsz_row = kHQ * splits * (kD + 2);
  const float* wrow = ws + ((int64_t)r * q_rows + row) * wsz_row;
  for (int i = tid; i < kHQ * (kD + 2); i += 192) {
    const int hq = i / (kD + 2);
    const int dd = i % (kD + 2);
    const float* hq_base = wrow + hq * splits * (kD + 2);
    float M = -INFINITY;
    for (int s = 0; s < splits; s++)
      M = fmaxf(M, hq_base[s * (kD + 2) + kD]);
    float l = 0.0f, o = 0.0f;
    if (M != -INFINITY) {
      for (int s = 0; s < splits; s++) {
        const float* w = hq_base + s * (kD + 2);
        const float w_s = expf(w[kD] - M);
        l += w_s * w[kD + 1];
        if (dd < kD) o += w_s * w[dd];
      }
    }
    if (dd < kD)
      out[(((int64_t)r * q_rows + row) * kHQ + hq) * kD + dd] =
          __float2half((M != -INFINITY && l > 0.0f) ? o / l : 0.0f);
  }
}

// Dynamic shared-memory footprint of one CTA of the given geometry.
template <int QROWS>
constexpr int smem_bytes() {
  constexpr int kMR = QROWS * kHQ;
  return kMR * kDS * static_cast<int>(sizeof(__half)) +
         2 * kN * kDS * static_cast<int>(sizeof(__half)) +
         kMR * (kN + 4) * static_cast<int>(sizeof(float)) +
         kMR * kN * static_cast<int>(sizeof(__half)) +
         3 * kMR * static_cast<int>(sizeof(float));
}

}  // namespace

namespace {

#define FMHA_LAUNCH(Q, MB, P, A)                                               \
  do {                                                                         \
    constexpr int kSB = smem_bytes<Q>();                                       \
    cudaFuncSetAttribute(fmha_prefill_wmma_kernel<Q, MB, P, A>,                 \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kSB);     \
    dim3 grid(static_cast<unsigned>(splits), 1, gz);                           \
    fmha_prefill_wmma_kernel<Q, MB, P, A><<<grid, Q * kHQ * 4, kSB, stream>>>(\
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),               \
        reinterpret_cast<const __half*>(kv_cache.data_ptr<at::Half>()),        \
        block_table.data_ptr<int>(), kv_lens.data_ptr<int>(),                  \
        ws.data_ptr<float>(), static_cast<float>(sm_scale),                    \
        static_cast<int>(splits), static_cast<int>(q_rows), n_page, bt_cols,   \
        kv_s0);                                                                \
  } while (0)

torch::Tensor fmha_prefill_cfg(torch::Tensor q, torch::Tensor kv_cache,
                               torch::Tensor block_table, torch::Tensor kv_lens,
                               torch::Tensor out, torch::Tensor ws,
                               double sm_scale, int64_t splits, int64_t q_rows,
                               int64_t cta_rows, int64_t pipe, int64_t abl) {
  const int R = static_cast<int>(q.size(0));
  const int Q = static_cast<int>(q.size(1));
  const int n_page = static_cast<int>(kv_cache.size(2));
  const int bt_cols = static_cast<int>(block_table.size(1));
  const int64_t kv_s0 = kv_cache.size(1) * kv_cache.size(2) *
      kv_cache.size(3) * kv_cache.size(4);
  TORCH_CHECK(cta_rows == 16 || cta_rows == 32, "cta_rows must be 16 or 32");
  TORCH_CHECK(Q % cta_rows == 0, "q_rows must be a multiple of cta_rows");
  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned gz = static_cast<unsigned>(R * (Q / cta_rows));
  const bool pp = pipe != 0;
  TORCH_CHECK(abl >= 0 && abl <= 3, "abl must be 0..3");
  TORCH_CHECK(abl == 0 || (cta_rows == 16 && pp),
              "ablation builds exist only for cta_rows=16, pipe=1");
  if (cta_rows == 16) {
    if (!pp) {
      FMHA_LAUNCH(16, 2, 0, 0);
    } else if (abl == 1) {
      FMHA_LAUNCH(16, 2, 1, 1);
    } else if (abl == 2) {
      FMHA_LAUNCH(16, 2, 1, 2);
    } else if (abl == 3) {
      FMHA_LAUNCH(16, 2, 1, 3);
    } else {
      FMHA_LAUNCH(16, 2, 1, 0);
    }
  } else {
    FMHA_LAUNCH(32, 1, 1, 0);
  }
  fmha_prefill_merge_kernel<<<static_cast<unsigned>(R * Q), 192, 0, stream>>>(
      ws.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      static_cast<int>(splits), static_cast<int>(q_rows));
  return out;
}

#undef FMHA_LAUNCH

}  // namespace

torch::Tensor fmha_prefill(torch::Tensor q, torch::Tensor kv_cache,
                           torch::Tensor block_table, torch::Tensor kv_lens,
                           torch::Tensor out, torch::Tensor ws,
                           double sm_scale, int64_t splits, int64_t q_rows) {
  return fmha_prefill_cfg(q, kv_cache, block_table, kv_lens, out, ws,
                          sm_scale, splits, q_rows, 16, 0, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fmha_prefill", &fmha_prefill,
        "T12 sm70 TC dense prefill attention (offline gate)");
  m.def("fmha_prefill_cfg", &fmha_prefill_cfg,
        "T12 sm70 TC dense prefill attention, tiled CTA M-rows + optional "
        "software-pipelined staging (b-1)",
        py::arg("q"), py::arg("kv_cache"), py::arg("block_table"),
        py::arg("kv_lens"), py::arg("out"), py::arg("ws"), py::arg("sm_scale"),
        py::arg("splits"), py::arg("q_rows"), py::arg("cta_rows"),
        py::arg("pipe") = 0, py::arg("abl") = 0);
}
