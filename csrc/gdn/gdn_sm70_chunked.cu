// SM70 (V100) GDN chunked (FLA-style) forward — P3 shadow kernel.
//
// Pure-FMA fp32-math kernels (no WMMA: total work per call is a few GFLOP;
// V100 fp32 FMA is sufficient and keeps the state in clean fp32, same as the
// incumbent gdn_sm70_fwd.cu). See /data/src/sm70_docs/p3_gdn_math_sheet.md
// for the FLA-derived math and /data/src/sm70_docs/2026-10-14-p3-gdn-shadow-
// kernel-spec.md for gates/seam.
//
// Pipeline (chunk C=64, NT = ceil(T/C)):
//   K1  WY pass     grid=(NB, N*Hv): per (n,hv,chunk)
//         g_cs = cumsum(g in chunk)                       (P1)
//         A = beta_t * exp(gcs_t-gcs_j) * k_t.k_j, t>j     (P2 kkt, strict lower)
//         A = (I+A)^{-1}  forward substitution in smem    (P2 solve)
//         u = A.(beta*v)   w = A.(beta*k*exp(gcs))         (P2 wy)
//   K2  state scan  grid=(V/32, N*Hv): per (n,hv) serial over NT
//         store h_c ; v_new = u_c - w_c h_c^T  (pre-decay, stored for K3)
//         h = h*exp(g_last) + k_c^T . (v_new * exp(g_last-gcs_t))
//   K3  output      grid=(V/32, NB, N*Hv): per (n,hv,chunk)
//         o = scale * ( q h_c^T * exp(gcs) + tril(qk^T)*exp(gcs diff) . v_new )
//
// Math source: vllm/third_party/flash_linear_attention/ops/ (FLA) — the
// incumbent token-loop kernel is the behavioral ground truth (parity test
// sm70_tests/test_gdn_chunked_parity.py, FP64 reference from its semantics).
//
// Env gate lives in sm70_gdn.py (VLLM_SM70_GDN_CHUNKED, default 0 = incumbent).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <vector>

namespace {

constexpr int C = 64;       // FLA chunk size
constexpr int K_DIM = 128;  // head dim K
constexpr int V_DIM = 128;  // head dim V
constexpr int BV2 = 16;     // K2 value tile (v4: 32->16: 24->48 CTAs, latency
                            // spread; NCU occ 12.5%, grid-too-small OPT.
                            // BV2=8 probed: 96 CTAs > 80 SMs -> K2 3.1->4.5ms,
                            // reverted)
constexpr int BV3 = 32;     // K3 value tile — kept 32: BV=16 doubled K3's grid
                            // (1 CTA/SM) and cut per-thread ILP 8->4: K3
                            // regressed 3.7 -> 4.2ms

__device__ __forceinline__ float h2f(const __half h) {
  return __half2float(h);
}

// ---------------------------------------------------------------------------
// Device-side chunk offsets (avoids a host sync per call).
// cu_seqlens [N+1] int32 -> chunk_offsets [N+1] int32 (global chunk ids).
// ---------------------------------------------------------------------------
__global__ void gdn_chunk_offsets_kernel(const int32_t* cu_seqlens,
                                         int32_t* chunk_offsets, int n) {
  if (threadIdx.x == 0) {
    int off = 0;
    for (int i = 0; i < n; ++i) {
      chunk_offsets[i] = off;
      off += (int)((size_t)(cu_seqlens[i + 1] - cu_seqlens[i]) + C - 1) / C;
    }
    chunk_offsets[n] = off;
  }
}

// ---------------------------------------------------------------------------
// K1: WY pass. grid=(NB, N*Hv), block=(256). smem: kS/vS (C x K half, reused)
// + A (C x C fp32) + gcs/beta (C fp32) ~= 34KB.
// ---------------------------------------------------------------------------
__global__ void gdn_wy_k1_kernel(
    const __half* __restrict__ qk,   // [1,T,Hq,K] (k used; q not)
    const __half* __restrict__ v,    // [1,T,Hv,V]
    const float* __restrict__ g,     // [1,T,Hv]
    const float* __restrict__ beta,  // [1,T,Hv]
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    __half* __restrict__ u,         // [1,T,Hv,V]
    __half* __restrict__ w,         // [1,T,Hv,K]
    float* __restrict__ gcs,        // [1,T,Hv]
    int q_heads, int v_heads, int nb, int n) {
  const int gc = blockIdx.x;
  const int nh = blockIdx.y;
  const int hv = nh % v_heads;
  const int nidx = nh / v_heads;
  const int hq = hv / (v_heads / q_heads);

  // v4 (blade 4): pad-2 rows. NCU (gdn_b3.ncu-rep) flagged 71% of K1's shared
  // wavefronts as excessive: mat's 256B row stride (128 half) made the kkt
  // dot's 8-row reads a 4-way conflict; A's 256B stride (64 float) the same
  // for the u/w GEMMs' A[r0][j] reads (8-way pattern). +2 -> 260B/264B rows
  // = +4B/+8B bank offset per row -> conflict-free.
  __shared__ __half mat[C][K_DIM + 2];   // k then v (reused), 16.6KB
  __shared__ float A[C][C + 2];          // 16.9KB
  __shared__ float gcs_s[C], beta_s[C];
  // v5: per-row scale factor folded into staging (was re-multiplied in the
  // GEMM inner loops: 524K redundant smem reads of beta_s per CTA, plus
  // 524K per-element expf in the w GEMM)
  __shared__ float wscale[C];

  // find sequence n containing global chunk gc
  int seq = -1;
  {
    int start = 0;
    // i < n: the i == n iteration is dead (empty window: start == end) and
    // reads cu_seqlens[n+1], one int past the end of the buffer (OOB read).
    for (int i = 0; i < n; ++i) {
      if (i > 0) start = chunk_offsets[i];
      if (gc < chunk_offsets[i] + (cu_seqlens[i + 1] - cu_seqlens[i] + C - 1) / C &&
          gc >= start) { seq = i; break; }
    }
  }
  if (seq < 0) return;  // beyond last real chunk
  const int bos = cu_seqlens[seq];
  const int seq_len = cu_seqlens[seq + 1] - bos;
  const int local_t = gc - chunk_offsets[seq];
  const int t0 = local_t * C;
  if (t0 >= seq_len) return;
  const int t1 = min(t0 + C, seq_len);

  const int t = threadIdx.x;
  // load g/beta for this chunk
  if (t < C) {
    const bool live = t < t1 - t0;
    const int64_t gi = (int64_t)(bos + t0 + t) * v_heads + hv;
    gcs_s[t] = live ? g[gi] : 0.0f;
    beta_s[t] = live ? beta[gi] : 0.0f;
  }
  __syncthreads();
  // inclusive cumsum (thread 0, C=64 is cheap)
  if (t == 0) {
    float acc = 0.0f;
    for (int i = 0; i < C; ++i) { acc += gcs_s[i]; gcs_s[i] = acc; }
  }
  // stage k chunk into mat[]
  {
    const __half* ksrc =
        qk + ((int64_t)(bos + t0) * q_heads + hq) * K_DIM;
    for (int i = t; i < C * K_DIM; i += 256) {
      const int r = i / K_DIM, c = i % K_DIM;
      mat[r][c] = (r < t1 - t0) ? ksrc[(int64_t)r * q_heads * K_DIM + c]
                                : __float2half(0.0f);
    }
  }
  __syncthreads();

  // A[t][j] = (t>j) * beta_t * exp(gcs_t - gcs_j) * sum_k k[t,k] k[j,k]
  // 256 threads, each owns a 4x4 block of (t,j)
  {
    const int bt = t / 16, bc = t % 16;
    const int t_base = bt * 4, j_base = bc * 4;
    for (int dt = 0; dt < 4; ++dt) {
      for (int dj = 0; dj < 4; ++dj) {
        const int r = t_base + dt, col = j_base + dj;
        float acc = 0.0f;
        if (r > col && r < t1 - t0 && col < t1 - t0) {
          for (int kk = 0; kk < K_DIM; ++kk)
            acc += h2f(mat[r][kk]) * h2f(mat[col][kk]);
          acc *= beta_s[r] * expf(gcs_s[r] - gcs_s[col]);
        }
        A[r][col] = acc;  // explicit 0 for masked (upper-tri/pad) entries
      }
    }
  }
  __syncthreads();

  // (Candidate: no explicit inverse. The two systems (I+A)u = beta*v and
  //  (I+A)w = beta*k*exp(g) are solved by the in-place blocked TRSM below.
  //  Removes the 64-step serial substitution, its 64 barriers and the
  //  local-memory xs[]: 9.58 -> 2.73 ms on the shape prototype.)

  // u = A.(beta*v) ; w = A.(beta*k*exp(gcs)) ; then store u (v restaged)
  // re-stage v over mat[], pre-scaled by beta (v5: GEMM inner loop is pure FMA)
  {
    const __half* vsrc =
        v + ((int64_t)(bos + t0) * v_heads + hv) * V_DIM;
    if (t < C) wscale[t] = beta_s[t];
    __syncthreads();
    for (int i = t; i < C * V_DIM; i += 256) {
      const int r = i / V_DIM, c = i % V_DIM;
      mat[r][c] = (r < t1 - t0)
                       ? __float2half(
                             h2f(vsrc[(int64_t)r * v_heads * V_DIM + c]) *
                             wscale[r])
                       : __float2half(0.0f);
    }
  }
  __syncthreads();
  {
    // Blocked TRSM: solve (I + A).X = mat IN PLACE, one thread per RHS column
    // (columns are independent => no barrier inside the solve). Row blocks of
    // 16: the within-block unit-lower chain stays in registers; already-solved
    // rows are read back from mat[] (in-place), so no extra smem is used.
    const int rc = t;
    if (rc < V_DIM) {
      float o[16];
#pragma unroll 1
      for (int b = 0; b < C / 16; ++b) {
        float res[16];
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const int r = b * 16 + q;
          float acc = (r < t1 - t0) ? h2f(mat[r][rc]) : 0.0f;
#pragma unroll 4
          for (int m = 0; m < b * 16; ++m) acc -= A[r][m] * h2f(mat[m][rc]);
          res[q] = acc;
        }
#pragma unroll
        for (int q = 0; q < 16; ++q) {
#pragma unroll
          for (int m = 0; m < q; ++m)
            res[q] -= A[b * 16 + q][b * 16 + m] * o[m];
          o[q] = res[q];
          const int r = b * 16 + q;
          if (r < t1 - t0) {
            mat[r][rc] = __float2half(o[q]);
            u[((int64_t)(bos + t0 + r) * v_heads + hv) * V_DIM + rc] =
                __float2half(o[q]);
          }
        }
      }
    }
  }
  // Barrier REQUIRED: the w-pass below re-stages k over mat[] (overwriting),
  // while the u GEMM above still reads mat[]. Without it, threads with
  // r0 >= len (skip the u GEMM on partial chunks) race ahead and clobber
  // live v rows mid-read.
  __syncthreads();
  // w = A.(beta*k*exp(gcs)) : re-stage k over mat[], pre-scaled by
  // beta*exp(gcs) (v5: 64 expf total instead of 524K per-element)
  {
    if (t < C) wscale[t] = beta_s[t] * expf(gcs_s[t]);
    __syncthreads();
    const __half* ksrc =
        qk + ((int64_t)(bos + t0) * q_heads + hq) * K_DIM;
    for (int i = t; i < C * K_DIM; i += 256) {
      const int r = i / K_DIM, c = i % K_DIM;
      mat[r][c] = (r < t1 - t0)
                       ? __float2half(
                             h2f(ksrc[(int64_t)r * q_heads * K_DIM + c]) *
                             wscale[r])
                       : __float2half(0.0f);
    }
  }
  __syncthreads();
  {
    // Same in-place blocked TRSM as above, second RHS.
    const int rc = t;
    if (rc < K_DIM) {
      float o[16];
#pragma unroll 1
      for (int b = 0; b < C / 16; ++b) {
        float res[16];
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const int r = b * 16 + q;
          float acc = (r < t1 - t0) ? h2f(mat[r][rc]) : 0.0f;
#pragma unroll 4
          for (int m = 0; m < b * 16; ++m) acc -= A[r][m] * h2f(mat[m][rc]);
          res[q] = acc;
        }
#pragma unroll
        for (int q = 0; q < 16; ++q) {
#pragma unroll
          for (int m = 0; m < q; ++m)
            res[q] -= A[b * 16 + q][b * 16 + m] * o[m];
          o[q] = res[q];
          const int r = b * 16 + q;
          if (r < t1 - t0) {
            mat[r][rc] = __float2half(o[q]);
            w[((int64_t)(bos + t0 + r) * v_heads + hv) * K_DIM + rc] =
                __float2half(o[q]);
          }
        }
      }
    }
  }
  // store gcs (only live rows — buffer has T rows, not C)
  if (t < t1 - t0) {
    const int64_t gi = (int64_t)(bos + t0 + t) * v_heads + hv;
    gcs[gi] = gcs_s[t];
  }
}

// ---------------------------------------------------------------------------
// K2: state scan. grid=(V/BV2, N*Hv), block=(512) (v5: 16 warps for latency hiding).
// v2 (blade 2): h lives in smem TRANSPOSED (K-major, pad-2: hT[k][v]) —
// v1's h[v][k] (128-float row stride, 128 ≡ 0 mod 32 banks) made every
// vnew/hupdate smem access a 32-way bank conflict (measured v1: 14.4ms vs
// ~0.1ms FMA floor = ~1400x; the conflict, not FMA, was the cost). K-major
// + pad-2: dot reads are stride-1 across banks, the transposed h_ws store
// and final-state store are conflict-free, and the h decay is fused into the
// h-update (single owner write, one fewer barrier than v1).
// smem: kS (C x K half, 16KB) + wS (C x K half, 16KB) + uS (C x BV2 half, 4KB)
//       + gcs (C fp32) + dynamic: hT (K x (BV2+2) fp32, 17.4KB)
//              + vnewS (C x BV2 fp32, 8KB) = 25.6KB
// ---------------------------------------------------------------------------
// k2 block size: parameterized (was hard-coded 512). 1cat's GDN prefill
// campaign measured fewer threads beating the old count ~1.5x on a similar
// shape; K2_NT keeps every row-group formula consistent.
constexpr int K2_NT = 512;
__global__ void gdn_state_k2_kernel(
    const __half* __restrict__ u,    // [1,T,Hv,V]
    const __half* __restrict__ w,    // [1,T,Hv,K]
    const __half* __restrict__ k,    // [1,T,Hq,K]
    const float* __restrict__ gcs,   // [1,T,Hv]
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    const float* __restrict__ initial_state,  // [N,Hv,V,K] or null
    __half* __restrict__ vnew,    // [1,T,Hv,V]
    float* __restrict__ h_ws,      // [NB,N*Hv,V,K] (state at chunk starts)
    float* __restrict__ final_state,  // [N,Hv,V,K] or null
    int q_heads, int v_heads, int nb, int n) {
  const int vt = blockIdx.x;
  const int nh = blockIdx.y;
  const int hv = nh % v_heads;
  const int hq = hv / (v_heads / q_heads);
  const int nidx = nh / v_heads;
  const int nt = chunk_offsets[nidx + 1] - chunk_offsets[nidx];
  if (nt == 0) return;

  // dynamic smem (25.6KB): hT (K x (BV2+2) fp32, 17.4KB) then
  // vnewS (C x BV2 fp32, 8KB)
  extern __shared__ float dyn_s[];
  float (*hT)[BV2 + 2] = (float (*)[BV2 + 2])dyn_s;          // K_DIM x (BV2+2)
  // v3 (blade 3): vnewS pad-2 and kST (k transposed, pad-2) — blade-2 left
  // vnewS [C x BV2] (32-float row = 128B ≡ 0 mod 32 banks) and kS [C x K]
  // (256B row, same) conflict-free only on the vnew-dot side; the h-update
  // r-loop reads vnewS[r][cc]/kS[r][kk] with FIXED col, varying row = 32-way
  // on both (measured blade-2: K2 7.7ms, ~1000x the FMA floor).
  float (*vnewS)[BV2 + 2] =
      (float (*)[BV2 + 2])(dyn_s + K_DIM * (BV2 + 2));
  __shared__ __half kST[K_DIM][C + 2];   // 16KB + pad — k staged K-major
  __shared__ __half wS[C][K_DIM + 2];    // 16KB + pad — full K (dot over all K)
  __shared__ __half uS[C][BV2 + 2];       // 4KB + pad
  __shared__ float gcs_s[C];
  __shared__ float g_last_s;
  // v3 (blade 3b): precomputed exp(g_last - gcs) per row — blade-2's step-4
  // did 4096 expf() per chunk step (~35us/step on V100 software expf; the
  // whole K2 7.7ms was dominated by it, not the dots).
  __shared__ float vdecay_s[C];
  // v4: BV2=16 -> static ~36KB + dynamic 14KB = ~50KB (< 64KB cc7.0 block cap)
  const int v0 = vt * BV2;

  const int t = threadIdx.x;
  // h init (transposed write: hT[k][r] = initial_state[v0 + r][k])
  for (int i = t; i < BV2 * K_DIM; i += K2_NT) {
    const int r = i / K_DIM, c = i % K_DIM;
    hT[c][r] = 0.0f;
    if (initial_state != nullptr) {
      const int64_t idx =
          ((int64_t)nidx * v_heads + hv) * V_DIM * K_DIM +
          (int64_t)(v0 + r) * K_DIM + c;
      hT[c][r] = initial_state[idx];
    }
  }
  __syncthreads();

  const int bos = cu_seqlens[nidx];
  for (int cch = 0; cch < nt; ++cch) {
    const int row0 = cch * C;
    const int len = min(C, cu_seqlens[nidx + 1] - bos - row0);

    // 1. store h_c to h_ws (row-major [V,K]; transposed read of hT —
    // conflict-free: for fixed r, bank = (34*cc + r) mod 32 = 2cc + r)
    __syncthreads();  // hT finalized by the previous step's h-update
    {
      float* hd = h_ws + ((int64_t)(chunk_offsets[nidx] + cch) * v_heads + hv) *
                          V_DIM * K_DIM;
      for (int i = t; i < BV2 * K_DIM; i += K2_NT) {
        const int r = i / K_DIM, cc = i % K_DIM;
        hd[(int64_t)(v0 + r) * K_DIM + cc] = hT[cc][r];
      }
    }
    // 2. load w_c, u_c (BV2 slice), k_c, gcs
    {
      const __half* wsrc =
          w + ((int64_t)(bos + row0) * v_heads + hv) * K_DIM;
      const __half* usrc =
          u + ((int64_t)(bos + row0) * v_heads + hv) * V_DIM;
      const __half* ksrc =
          k + ((int64_t)(bos + row0) * q_heads + hq) * K_DIM;
      for (int i = t; i < C * BV2; i += K2_NT) {
        const int r = i / BV2, cc = i % BV2;
        uS[r][cc] = (r < len)
                        ? usrc[(int64_t)r * v_heads * V_DIM + v0 + cc]
                        : __float2half(0.0f);
      }
      for (int i = t; i < C * K_DIM; i += K2_NT) {
        const int r = i / K_DIM, cc = i % K_DIM;
        wS[r][cc] = (r < len)
                        ? wsrc[(int64_t)r * v_heads * K_DIM + cc]
                        : __float2half(0.0f);
      }
      // v3 (blade 3): stage k TRANSPOSED (kST[kk][r], row stride C+2 halves
      // = 132B = +4B bank offset per row) — conflict-free on both the write
      // (warp covers 32 consecutive kk at fixed r) and the h-update r-loop
      // (fixed kk, varying r).
      for (int i = t; i < C * K_DIM; i += K2_NT) {
        const int r = i / K_DIM, cc = i % K_DIM;
        kST[cc][r] = (r < len)
                        ? ksrc[(int64_t)r * q_heads * K_DIM + cc]
                        : __float2half(0.0f);
      }
      if (t < C)
        gcs_s[t] = (t < len)
                        ? gcs[(int64_t)(bos + row0 + t) * v_heads + hv]
                        : 0.0f;
    }
    __syncthreads();
    if (t == 0) g_last_s = gcs_s[len - 1];
    __syncthreads();
    // v3 (blade 3b): 64 parallel expf per step (was 4096 per-element in the
    // row-scale loop) — V100 software expf is the per-step cost, not FMA.
    if (t < C) vdecay_s[t] = (t < len) ? expf(g_last_s - gcs_s[t]) : 0.0f;
    __syncthreads();

    // 3. vnew = u - w.h^T   [C x BV2]; v4 (BV2=16): 256 thr = 16 cols x 16
    // row-groups (4 rows each). Warp pattern (16 c x 2 rows) stays
    // conflict-free on wS (260B row stride) and hT (136B row stride).
    {
      const int c0 = t % BV2, rg = t / BV2;
#pragma unroll
      for (int dt = 0; dt < C / (K2_NT / BV2); ++dt) {
        const int r0 = rg * (C / (K2_NT / BV2)) + dt;
        float acc = 0.0f;
        if (r0 < len) {
          for (int kk = 0; kk < K_DIM; kk += 2)
            acc += h2f(wS[r0][kk]) * hT[kk][c0] +
                   h2f(wS[r0][kk + 1]) * hT[kk + 1][c0];
        }
        vnewS[r0][c0] = (r0 < len) ? h2f(uS[r0][c0]) - acc : 0.0f;
      }
    }
    __syncthreads();
    // store vnew (pre-decay) to global. (v5c prefetch variant was
    // negative: moving w/u/gcs staging into this window delayed the hT
    // update and lengthened the next step's critical path — G2 0.593
    // -> 0.626. Archived, reverted.)
    {
      __half* vdst =
          vnew + ((int64_t)(bos + row0) * v_heads + hv) * V_DIM;
      for (int i = t; i < C * BV2; i += K2_NT) {
        const int r = i / BV2, cc = i % BV2;
        if (r < len)
          vdst[(int64_t)r * v_heads * V_DIM + v0 + cc] =
              __float2half(vnewS[r][cc]);
      }
    }
    // 4. row-scale vnew' = vnew .* exp(g_last - gcs)  (blade 3b: precomputed
    // per-row factor, no per-element transcendental)
    for (int i = t; i < C * BV2; i += K2_NT) {
      const int r = i / BV2, cc = i % BV2;
      vnewS[r][cc] = (r < len) ? vnewS[r][cc] * vdecay_s[r] : 0.0f;
    }
    __syncthreads();
    // 5. h = h*exp(g_last) + k^T . vnew'   (decay FUSED into the update: one
    // owner write per element, no separate decay pass)
    // v3 (blade 3): 4 split accumulators (ILP 4; blade-2's `unroll 1` r-loop
    // was a 64-deep dependent FMA chain at ~1 cycle/FMA) + conflict-free
    // kST/vnewS (see staging comments).
    {
      const float decay = expf(g_last_s);
      // v4 (BV2-generic): thread owns col cc, kk in K_DIM/per groups.
      // BV2=16: 16 cols x 16 k-groups x 8 steps; BV2=8: 8 x 32 x 4.
      const int per = K2_NT / BV2;
      const int cc = t / per;
      // kk0 stride 1 (not K_DIM/per) keeps the 32 lanes of a warp on 32
      // distinct banks of kST: bank = (kk + r/2) % 32 with row stride 66
      // halves, so consecutive kk is conflict-free while a stride of
      // K_DIM/per collapses to 8 banks (4-way conflicted, measured).
      const int kk0 = t % per;
      // r-outer: vnewS[r][cc] is reused across all dk (one broadcast read
      // instead of K_DIM/per); per-CTA smem loads 512 -> 320 (ncu:
      // mio_throttle 23.6% with the dk-outer form). Per-kk r-sum order
      // changes (tree of 4 splits -> running sum): fp32-rounding level only.
      constexpr int NDK = K_DIM / per;
      float h0[NDK], acc[NDK] = {0.f};
#pragma unroll
      for (int dk = 0; dk < NDK; ++dk)
        h0[dk] = hT[kk0 + dk * per][cc] * decay;
      for (int r = 0; r < C; r += 4) {
        const float v0 = vnewS[r][cc], v1 = vnewS[r + 1][cc];
        const float v2 = vnewS[r + 2][cc], v3 = vnewS[r + 3][cc];
#pragma unroll
        for (int dk = 0; dk < NDK; ++dk) {
          const __half* ks = &kST[kk0 + dk * per][r];
          acc[dk] += v0 * h2f(ks[0]) + v1 * h2f(ks[1]);
          acc[dk] += v2 * h2f(ks[2]) + v3 * h2f(ks[3]);
        }
      }
#pragma unroll
      for (int dk = 0; dk < NDK; ++dk)
        hT[kk0 + dk * per][cc] = h0[dk] + acc[dk];
    }
    // (the barrier at the top of the next iteration guards the hT read)
  }

  // final state (transposed read of hT)
  __syncthreads();
  if (final_state != nullptr) {
    float* fd =
        final_state +
        ((int64_t)nidx * v_heads + hv) * V_DIM * K_DIM;
    for (int i = t; i < BV2 * K_DIM; i += K2_NT) {
      const int r = i / K_DIM, cc = i % K_DIM;
      fd[(int64_t)(v0 + r) * K_DIM + cc] = hT[cc][r];
    }
  }
}

// ---------------------------------------------------------------------------
// K3: output. grid=(V/BV3, NB, N*Hv), block=(256).
// smem: h (BV3 x K half) + kS (C x K half) + qS (C x K half) + vnewS (C x BV3 half)
//       + P (C x C fp32) + gcs (C fp32) ~= 8+16+16+4+16+0.25 = 60KB (opt-in)
// ---------------------------------------------------------------------------
__global__ void gdn_output_k3_kernel(
    const __half* __restrict__ q,     // [1,T,Hq,K]
    const __half* __restrict__ k,     // [1,T,Hq,K]
    const __half* __restrict__ vnew,  // [1,T,Hv,V]
    const float* __restrict__ gcs,    // [1,T,Hv]
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    const float* __restrict__ h_ws,   // [NB,N*Hv,V,K]
    __half* __restrict__ out,         // [1,T,Hv,V]
    float scale, int q_heads, int v_heads, int nb, int n) {
  const int vt = blockIdx.x;
  const int gc = blockIdx.y;
  const int nh = blockIdx.z;
  const int hv = nh % v_heads;
  const int nidx = nh / v_heads;
  const int hq = hv / (v_heads / q_heads);

  // locate sequence for gc
  int seq = -1;
  {
    int start = 0;
    // i < n: the i == n iteration is dead (empty window) and reads
    // cu_seqlens[n+1], one int past the end of the buffer (OOB read).
    for (int i = 0; i < n; ++i) {
      if (i > 0) start = chunk_offsets[i];
      const int end =
          chunk_offsets[i] +
          (cu_seqlens[i + 1] - cu_seqlens[i] + C - 1) / C;
      if (gc >= start && gc < end) { seq = i; break; }
    }
  }
  if (seq < 0) return;
  const int bos = cu_seqlens[seq];
  const int seq_len = cu_seqlens[seq + 1] - bos;
  const int t0 = (gc - chunk_offsets[seq]) * C;
  if (t0 >= seq_len) return;
  const int len = min(C, seq_len - t0);
  const int v0 = vt * BV3;

  // v4: pad-2 on the 256B-stride rows (hS/kS/qS 128-half = 256B ≡ 0 mod 32
  // banks; P 64-float = 256B) — NCU showed 36% excessive shared wavefronts
  // in K2 and K1's u/w GEMM pattern repeats here (P dot, state dot, P read).
  __shared__ __half hS[BV3][K_DIM + 2];    // 4.2KB
  __shared__ __half kS[C][K_DIM + 2];     // 16.6KB
  __shared__ __half qS[C][K_DIM + 2];     // 16.6KB
  __shared__ __half vnS[C][BV3 + 2];       // 2.3KB
  __shared__ float gcs_s[C];
  // dynamic smem (16.9KB): P[C][C+2] fp32
  extern __shared__ float dyn_s3[];
  float (*P)[C + 2] = (float (*)[C + 2])dyn_s3;  // 4224 floats

  const int t = threadIdx.x;
  {
    const float* hs =
        h_ws + ((int64_t)gc * v_heads + hv) * V_DIM * K_DIM;
    const __half* ks = k + ((int64_t)(bos + t0) * q_heads + hq) * K_DIM;
    const __half* qs = q + ((int64_t)(bos + t0) * q_heads + hq) * K_DIM;
    const __half* vns = vnew + ((int64_t)(bos + t0) * v_heads + hv) * V_DIM;
    for (int i = t; i < BV3 * K_DIM; i += 256) {
      const int r = i / K_DIM, c = i % K_DIM;
      hS[r][c] = __float2half(hs[(int64_t)(v0 + r) * K_DIM + c]);
    }
    for (int i = t; i < C * K_DIM; i += 256) {
      const int r = i / K_DIM, c = i % K_DIM;
      const bool live = r < len;
      const int64_t gidx = (int64_t)r * q_heads * K_DIM + c;
      kS[r][c] = live ? ks[gidx] : __float2half(0.0f);
      qS[r][c] = live ? qs[gidx] : __float2half(0.0f);
    }
    for (int i = t; i < C * BV3; i += 256) {
      const int r = i / BV3, c = i % BV3;
      vnS[r][c] = (r < len)
                      ? vns[(int64_t)r * v_heads * V_DIM + v0 + c]
                      : __float2half(0.0f);
    }
    if (t < C)
      gcs_s[t] = (t < len)
                     ? gcs[(int64_t)(bos + t0 + t) * v_heads + hv]
                     : 0.0f;
  }
  __syncthreads();

  // exp(+gcs) is hoisted for the state term below, where underflow to 0 is the
  // correct value. The pairwise decay is NOT factorised any more:
  // exp(gcs_t)*exp(-gcs_j) overflows to inf once gcs < -88, and real gates
  // accumulate to ~-830 inside a 64-token chunk (g <= 0 every token), so
  // egp*egm = 0*inf = NaN. Use the difference exp(gcs_t - gcs_j) <= 1 and skip
  // the vanishing band cheaply (gcs is monotone non-increasing).
  __shared__ float egp_s[C];  // exp(+gcs)
  for (int i = t; i < C; i += 256)
    egp_s[i] = expf(gcs_s[i]);
  __syncthreads();

  // P[t][j] = (t>=j) * exp(gcs_t - gcs_j) * sum_k q[t,k] k[j,k]
  {
    const int bt = t / 16, bc = t % 16;
    for (int dt = 0; dt < 4; ++dt) {
      const int r = bt * 4 + dt;
      for (int dj = 0; dj < 4; ++dj) {
        const int cj = bc * 4 + dj;
        float acc = 0.0f;
        if (r >= cj && r < len && cj < len) {
          for (int kk = 0; kk < K_DIM; kk += 2)
            acc += h2f(qS[r][kk]) * h2f(kS[cj][kk]) +
                   h2f(qS[r][kk + 1]) * h2f(kS[cj][kk + 1]);
          acc *= (gcs_s[r] - gcs_s[cj] < -88.0f)
                     ? 0.0f
                     : expf(gcs_s[r] - gcs_s[cj]);
        }
        P[r][cj] = acc;
      }
    }
  }
  __syncthreads();

  // o[r][c] = scale * ( sum_k q[r,k] h[c,k] * exp(gcs_r)
  //                    + sum_j P[r][j] * vnew[j][c] )
  // BV3=32: 256 thr = 64 rows x 32 cols, 4 thr/row x 8 cols (ILP 8 across the
  // col dim; the 1-col x 8-row variant of v4 cut ILP and regressed K3).
  {
    const int r0 = t / 4, c0 = (t % 4) * 8;
    if (r0 < len) {
      // exp(gcs_r) hoisted once (v1 recomputed it 8x per thread).
      const float state_exp = egp_s[r0];
      float state_acc[8];
#pragma unroll
      for (int dc = 0; dc < 8; ++dc) {
        const int c = c0 + dc;
        float acc = 0.0f;
        for (int kk = 0; kk < K_DIM; kk += 2)
          acc += h2f(qS[r0][kk]) * h2f(hS[c][kk]) +
                 h2f(qS[r0][kk + 1]) * h2f(hS[c][kk + 1]);
        state_acc[dc] = acc * state_exp;
      }
      float intra_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int j = 0; j <= r0 && j < len; ++j) {
        const float p = P[r0][j];
        // 8 contiguous vnS halves = 4 half2 (c0 is a multiple of 8 halves;
        // the pad-2 row stays 4B-aligned)
        const __half2* vn2 = (const __half2*)(vnS[j] + c0);
#pragma unroll
        for (int dc2 = 0; dc2 < 4; ++dc2) {
          const __half2 vp = vn2[dc2];
          intra_acc[dc2 * 2] += p * h2f(vp.x);
          intra_acc[dc2 * 2 + 1] += p * h2f(vp.y);
        }
      }
      // store: 8 contiguous halves = 4 half2
      __half2* op = (__half2*)
          (out + ((int64_t)(bos + t0 + r0) * v_heads + hv) * V_DIM + v0 + c0);
#pragma unroll
      for (int dc2 = 0; dc2 < 4; ++dc2)
        op[dc2] = __floats2half2_rn(
            (state_acc[dc2 * 2] + intra_acc[dc2 * 2]) * scale,
            (state_acc[dc2 * 2 + 1] + intra_acc[dc2 * 2 + 1]) * scale);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<torch::Tensor> gdn_chunked_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    c10::optional<torch::Tensor> initial_state,
    const torch::Tensor& cu_seqlens,
    double scale,
    bool output_final_state) {
  TORCH_CHECK(q.dtype() == torch::kHalf, "q must be fp16");
  const int T = q.size(1);
  const int q_heads = q.size(2);
  const int v_heads = v.size(2);
  const int V = v.size(3);
  TORCH_CHECK(v.size(2) == v_heads && v.size(3) == V);
  TORCH_CHECK(K_DIM == 128 && V_DIM == 128, "shadow kernel supports K=V=128");

  auto i32 = cu_seqlens.to(torch::kInt32).contiguous();
  const int64_t N = i32.numel() - 1;

  auto opts = q.options();
  auto i32_opts = opts.dtype(torch::kInt32);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int nb = (T + C - 1) / C;
  auto chunk_offsets = torch::empty({N + 1}, i32_opts);
  // Launch on the current stream (not default stream 0): K1/K2/K3 consume
  // chunk_offsets on `stream`, so the producer must be ordered with them
  // even when the current stream is not the legacy default.
  gdn_chunk_offsets_kernel<<<1, 1, 0, stream>>>(
      i32.data_ptr<int32_t>(), chunk_offsets.data_ptr<int32_t>(), (int)N);

  auto u = torch::empty({1, T, v_heads, V}, opts);
  auto w = torch::empty({1, T, v_heads, 128}, opts);
  auto gcs_ws = torch::empty({1, T, v_heads}, opts.dtype(torch::kFloat));
  auto vnew = torch::empty({1, T, v_heads, V}, opts);
  auto h_ws = torch::empty(
      {(int64_t)nb * N, v_heads, V_DIM, K_DIM},
      opts.dtype(torch::kFloat));
  auto out = torch::empty({1, T, v_heads, V}, opts);
  torch::Tensor final_state;
  if (output_final_state)
    final_state = torch::empty(
        {N, v_heads, V_DIM, K_DIM}, opts.dtype(torch::kFloat));

  static bool configured = false;
  // Offline attribution (JIT, env-gated; unset/"all" = production path):
  //   VLLM_SM70_GDN_PHASE=k1k3|k2|k1|k3  -> skip the others (timing only)
  //   VLLM_SM70_GDN_PROF=1               -> print per-phase CUDA-event ms
  const char* _ph = std::getenv("VLLM_SM70_GDN_PHASE");
  const std::string _phase = _ph ? _ph : "all";
  const bool _prof = std::getenv("VLLM_SM70_GDN_PROF") != nullptr;
  const bool run_k1 = _phase == "all" || _phase == "k1k3" || _phase == "k1";
  const bool run_k2 = _phase == "all" || _phase == "k2";
  const bool run_k3 = _phase == "all" || _phase == "k1k3" || _phase == "k3";
  cudaEvent_t _e0, _e1, _e2, _e3, _e4;
  if (_prof) { cudaEventCreate(&_e0); cudaEventCreate(&_e1);
               cudaEventCreate(&_e2); cudaEventCreate(&_e3);
               cudaEventCreate(&_e4);
               cudaEventRecord(_e0, stream); }

  if (!configured) {
    int optin_max = 0;
    cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           at::cuda::current_device());
    // attribute cap = optin_max - static smem; request only what K2/K3 need
    cudaError_t e1 = cudaFuncSetAttribute(gdn_state_k2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         32 * 1024);  // K2 uses 24KB
    cudaError_t e2 = cudaFuncSetAttribute(gdn_output_k3_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         20 * 1024);  // K3 uses 16KB
    TORCH_CHECK(e1 == cudaSuccess && e2 == cudaSuccess,
                "smem opt-in failed (optin_max=", optin_max, "B): ",
                cudaGetErrorName(e1), " / ", cudaGetErrorName(e2));
    TORCH_CHECK(optin_max >= 61 * 1024,
                "device smem opt-in too small for K2/K3: ", optin_max);
    configured = true;
  }
  auto chk = [&](const char* name) {
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, name, " CUDA error: ", cudaGetErrorString(e));
  };

  const __half* qp = reinterpret_cast<const __half*>(q.data_ptr());
  const __half* kp = reinterpret_cast<const __half*>(k.data_ptr());
  const __half* vp = reinterpret_cast<const __half*>(v.data_ptr());
  const float* gp = g.data_ptr<float>();
  const float* bp = beta.data_ptr<float>();
  const int32_t* cus = i32.data_ptr<int32_t>();
  const int32_t* cos = chunk_offsets.data_ptr<int32_t>();
  __half* up = reinterpret_cast<__half*>(u.data_ptr());
  __half* wp = reinterpret_cast<__half*>(w.data_ptr());
  float* gcp = gcs_ws.data_ptr<float>();
  __half* vnp = reinterpret_cast<__half*>(vnew.data_ptr());
  float* hwp = h_ws.data_ptr<float>();
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());

  {
    dim3 grid((unsigned)(nb * N), (unsigned)(N * v_heads));
    if (_prof) cudaEventRecord(_e1, stream);
    if (run_k1) gdn_wy_k1_kernel<<<grid, 256, 0, stream>>>(
        kp, vp, gp, bp, cus, cos, up, wp, gcp, q_heads, v_heads, nb,
        (int)N);
    chk("K1_wy");
  }
  {
    const float* h0p =
        initial_state.has_value()
            ? initial_state.value().data_ptr<float>()
            : nullptr;
    float* fsp = output_final_state ? final_state.data_ptr<float>() : nullptr;
    dim3 grid(V_DIM / BV2, (unsigned)(N * v_heads));
    if (_prof) cudaEventRecord(_e2, stream);
    if (run_k2) gdn_state_k2_kernel<<<grid, K2_NT, 14336, stream>>>(  // v4 BV=16: hT 9.2KB + vnewS 4.6KB
        up, wp, kp, gcp, cus, cos, h0p, vnp, hwp, fsp, q_heads, v_heads,
        nb, (int)N);
    chk("K2_state");
  }
  {
    dim3 grid(V_DIM / BV3, (unsigned)(nb * N), (unsigned)(N * v_heads));
    if (_prof) cudaEventRecord(_e3, stream);
    if (run_k3) gdn_output_k3_kernel<<<grid, 256, C * (C + 2) * 4, stream>>>(
        qp, kp, vnp, gcp, cus, cos, hwp, outp, (float)scale, q_heads,
        v_heads, nb, (int)N);
    if (_prof) {
      cudaEventRecord(_e4, stream); cudaEventSynchronize(_e4);
      float pa, pb, pc, pd;
      cudaEventElapsedTime(&pa, _e0, _e1); cudaEventElapsedTime(&pb, _e1, _e2);
      cudaEventElapsedTime(&pc, _e2, _e3); cudaEventElapsedTime(&pd, _e3, _e4);
      printf("[GDN-PHASE] pre=%.3f K1=%.3f K2=%.3f K3=%.3f ms\n", pa, pb, pc, pd);
    }
    chk("K3_output");
  }

  if (!output_final_state) final_state = torch::Tensor();
  return {out, final_state};
}

// debug: expose intermediates (u, w, gcs, vnew, h_ws, out, fs)
std::vector<torch::Tensor> gdn_chunked_forward_dbg(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    c10::optional<torch::Tensor> initial_state,
    const torch::Tensor& cu_seqlens,
    double scale) {
  const int T = q.size(1);
  const int q_heads = q.size(2);
  const int v_heads = v.size(2);
  const int V = v.size(3);
  auto i32 = cu_seqlens.to(torch::kInt32).contiguous();
  const int64_t N = i32.numel() - 1;
  auto opts = q.options();
  auto i32_opts = opts.dtype(torch::kInt32);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int nb = (T + C - 1) / C;
  auto chunk_offsets = torch::empty({N + 1}, i32_opts);
  // Launch on the current stream (not default stream 0): K1/K2/K3 consume
  // chunk_offsets on `stream`, so the producer must be ordered with them
  // even when the current stream is not the legacy default.
  gdn_chunk_offsets_kernel<<<1, 1, 0, stream>>>(
      i32.data_ptr<int32_t>(), chunk_offsets.data_ptr<int32_t>(), (int)N);

  auto u = torch::empty({1, T, v_heads, V}, opts);
  auto w = torch::empty({1, T, v_heads, 128}, opts);
  auto gcs_ws = torch::empty({1, T, v_heads}, opts.dtype(torch::kFloat));
  auto vnew = torch::empty({1, T, v_heads, V}, opts);
  auto h_ws = torch::empty(
      {(int64_t)nb * N, v_heads, V_DIM, K_DIM}, opts.dtype(torch::kFloat));
  auto out = torch::empty({1, T, v_heads, V}, opts);
  auto final_state = torch::empty(
      {N, v_heads, V_DIM, K_DIM}, opts.dtype(torch::kFloat));

    const char* _ph2 = std::getenv("VLLM_SM70_GDN_PHASE");
    const std::string _phase2 = _ph2 ? _ph2 : "all";
    const bool _prof2 = std::getenv("VLLM_SM70_GDN_PROF") != nullptr;
    const bool run2_k1 = _phase2 == "all" || _phase2 == "k1k3" || _phase2 == "k1";
    const bool run2_k2 = _phase2 == "all" || _phase2 == "k2";
    const bool run2_k3 = _phase2 == "all" || _phase2 == "k1k3" || _phase2 == "k3";
    cudaEvent_t _f0, _f1, _f2, _f3, _f4;
    if (_prof2) { cudaEventCreate(&_f0); cudaEventCreate(&_f1);
                  cudaEventCreate(&_f2); cudaEventCreate(&_f3);
                  cudaEventCreate(&_f4);
                  cudaEventRecord(_f0, stream); }
  {
    static bool cfg2 = false;

    if (!cfg2) {
      cudaFuncSetAttribute(gdn_state_k2_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           32 * 1024);
      cudaFuncSetAttribute(gdn_output_k3_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           20 * 1024);
      cfg2 = true;
    }
  }
  const __half* qp = reinterpret_cast<const __half*>(q.data_ptr());
  const __half* kp = reinterpret_cast<const __half*>(k.data_ptr());
  const __half* vp = reinterpret_cast<const __half*>(v.data_ptr());
  const float* gp = g.data_ptr<float>();
  const float* bp = beta.data_ptr<float>();
  const int32_t* cus = i32.data_ptr<int32_t>();
  const int32_t* cos = chunk_offsets.data_ptr<int32_t>();
  __half* up = reinterpret_cast<__half*>(u.data_ptr());
  __half* wp = reinterpret_cast<__half*>(w.data_ptr());
  float* gcp = gcs_ws.data_ptr<float>();
  __half* vnp = reinterpret_cast<__half*>(vnew.data_ptr());
  float* hwp = h_ws.data_ptr<float>();
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  const float* h0p = initial_state.has_value()
                        ? initial_state.value().data_ptr<float>()
                        : nullptr;
  float* fsp = final_state.data_ptr<float>();

  {
    dim3 grid((unsigned)(nb * N), (unsigned)(N * v_heads));
    if (_prof2) cudaEventRecord(_f1, stream);
    if (run2_k1) gdn_wy_k1_kernel<<<grid, 256, 0, stream>>>(
        kp, vp, gp, bp, cus, cos, up, wp, gcp, q_heads, v_heads, nb,
        (int)N);
  }
  {
    dim3 grid(V_DIM / BV2, (unsigned)(N * v_heads));
    if (_prof2) cudaEventRecord(_f2, stream);
    if (run2_k2) gdn_state_k2_kernel<<<grid, K2_NT, 14336, stream>>>(  // v4 BV=16: hT 9.2KB + vnewS 4.6KB
        up, wp, kp, gcp, cus, cos, h0p, vnp, hwp, fsp, q_heads, v_heads,
        nb, (int)N);
  }
  {
    dim3 grid(V_DIM / BV3, (unsigned)(nb * N), (unsigned)(N * v_heads));
    if (_prof2) cudaEventRecord(_f3, stream);
    if (run2_k3) gdn_output_k3_kernel<<<grid, 256, C * (C + 2) * 4, stream>>>(
        qp, kp, vnp, gcp, cus, cos, hwp, outp, (float)scale, q_heads,
        v_heads, nb, (int)N);
    if (_prof2) {
      cudaEventRecord(_f4, stream); cudaEventSynchronize(_f4);
      float qa, qb, qc, qd;
      cudaEventElapsedTime(&qa, _f0, _f1); cudaEventElapsedTime(&qb, _f1, _f2);
      cudaEventElapsedTime(&qc, _f2, _f3); cudaEventElapsedTime(&qd, _f3, _f4);
      printf("[GDN-PHASE] pre=%.3f K1=%.3f K2=%.3f K3=%.3f ms\n", qa, qb, qc, qd);
    }
  }
  return {u, w, gcs_ws, vnew, h_ws, out, final_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_chunked_forward", &gdn_chunked_forward,
        "SM70 GDN chunked (FLA-style) forward — P3 shadow kernel");
  m.def("gdn_chunked_forward_dbg", &gdn_chunked_forward_dbg,
        "debug: return intermediates (u,w,gcs,vnew,h_ws,out,fs)");
}
