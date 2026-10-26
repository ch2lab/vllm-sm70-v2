// SPDX-License-Identifier: Apache-2.0
// P1 QSA prefill — SM70 CUDA/WMMA prototype (T2.1: QK probe).
//
// Goal: validate that the nvcuda::wmma tensor-core path works on this box
// (sm_70, nvcc 12.8) and produces correct scores vs the scalar/FMA reference
// Triton path. S[m,n] = scale * sum_k Q[m,k] * K[n,k];  m<16, n<64, k<256.
//
// Isolation: new file, JIT-loaded by qsa_prefill_sm70.py behind an env gate;
// no change to the system main flow.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int kM = 16;    // WMMA m
constexpr int kN = 64;    // keys per CTA (4 n-tiles of 16)
constexpr int kD = 256;   // head_dim / contraction
constexpr int kThreads = 128;  // 4 warps, one per n-tile

__global__ void __launch_bounds__(kThreads)
wmma_qk_probe_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                     float* __restrict__ S, float scale) {
  __shared__ __align__(16) half sQ[kM * kD];
  __shared__ __align__(16) half sK[kN * kD];
  __shared__ __align__(16) float sS[kM * kN];

  const int tid = threadIdx.x;
  for (int i = tid; i < kM * kD; i += kThreads) sQ[i] = Q[i];
  for (int i = tid; i < kN * kD; i += kThreads) sK[i] = K[i];
  __syncthreads();

  const int warp = tid >> 5;  // 0..3 -> n-tile [warp*16, warp*16+16)

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
  wmma::fill_fragment(acc, 0.0f);
#pragma unroll
  for (int k0 = 0; k0 < kD; k0 += 16) {
    wmma::load_matrix_sync(af, sQ + k0, kD);
    // B(k,n) = K[n,k]; sK is [n][k] row-major, ld=kD -> col-major B.
    wmma::load_matrix_sync(bf, sK + warp * 16 * kD + k0, kD);
    wmma::mma_sync(acc, af, bf, acc);
  }
  // store this warp's 16x16 tile into sS[:, warp*16 : warp*16+16]
  wmma::store_matrix_sync(sS + warp * 16, acc, kN, wmma::mem_row_major);
  __syncthreads();

  for (int i = tid; i < kM * kN; i += kThreads) S[i] = sS[i] * scale;
}

constexpr int kA_M = kM;
constexpr int kA_N = kN;
constexpr int kA_D = kD;

__global__ void __launch_bounds__(kThreads)
wmma_attn_probe_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                       const half* __restrict__ V, half* __restrict__ O,
                       float scale) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  half* sQ = reinterpret_cast<half*>(smem_raw);
  half* sK = sQ + kA_M * kA_D;
  half* sV = sK + kA_N * kA_D;
  float* sS = reinterpret_cast<float*>(sV + kA_N * kA_D);
  half* sP = reinterpret_cast<half*>(sS + kA_M * kA_N);
  float* sT = reinterpret_cast<float*>(sP + kA_M * kA_N);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int i = tid; i < kA_M * kA_D; i += kThreads) sQ[i] = Q[i];
  for (int i = tid; i < kA_N * kA_D; i += kThreads) sK[i] = K[i];
  for (int i = tid; i < kA_N * kA_D; i += kThreads) sV[i] = V[i];
  __syncthreads();

  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int k0 = 0; k0 < kA_D; k0 += 16) {
      wmma::load_matrix_sync(af, sQ + k0, kA_D);
      wmma::load_matrix_sync(bf, sK + warp * 16 * kA_D + k0, kA_D);
      wmma::mma_sync(acc, af, bf, acc);
    }
    for (int i = 0; i < acc.num_elements; ++i) acc.x[i] *= scale;
    wmma::store_matrix_sync(sS + warp * 16, acc, kA_N, wmma::mem_row_major);
  }
  __syncthreads();

  if (tid < kA_M) {
    const int m = tid;
    float mx = -1e30f;
    for (int n = 0; n < kA_N; ++n) mx = fmaxf(mx, sS[m * kA_N + n]);
    float sum = 0.0f;
    for (int n = 0; n < kA_N; ++n) {
      float e = __expf(sS[m * kA_N + n] - mx);
      sP[m * kA_N + n] = __float2half(e);
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int n = 0; n < kA_N; ++n)
      sP[m * kA_N + n] = __float2half(__half2float(sP[m * kA_N + n]) * inv);
  }
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
#pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    const int d0 = (warp * 4 + dt) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int n0 = 0; n0 < kA_N; n0 += 16) {
      wmma::load_matrix_sync(pf, sP + n0, kA_N);
      wmma::load_matrix_sync(vf, sV + n0 * kA_D + d0, kA_D);
      wmma::mma_sync(acc, pf, vf, acc);
    }
    wmma::store_matrix_sync((sT + warp * 256), acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 16 * 16; i += 32) {
      const int m = i >> 4;
      const int d = i & 15;
      O[m * kA_D + d0 + d] = __float2half((sT + warp * 256)[i]);
    }
    __syncwarp();
  }
}

constexpr int kQ_M = 16;
constexpr int kQ_BN = 32;
constexpr int kQ_D = 256;
constexpr int kQ_T = 1024;
constexpr int kQ_PS = 784;

__device__ __forceinline__ float qsigmoid_(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// One CTA per query row. G q-heads (<=16), D=256, sparse paged KV.
__global__ void __launch_bounds__(kQ_T)
qsa_prefill_attn_kernel(const half* __restrict__ q,
                        const half* __restrict__ k_cache,
                        const half* __restrict__ v_cache,
                        const int* __restrict__ indices,
                        const int* __restrict__ block_table,
                        const int* __restrict__ token_to_req,
                        half* __restrict__ out, const half* __restrict__ gate,
                        int G, int TOPK, int page_width, int nreq, float scale) {
  extern __shared__ __align__(16) unsigned char raw[];
  half* sQ = reinterpret_cast<half*>(raw);
  half* sK = sQ + kQ_M * kQ_D;
  half* sV = sK + kQ_BN * kQ_D;
  float* sS = reinterpret_cast<float*>(sV + kQ_BN * kQ_D);
  half* sP = reinterpret_cast<half*>(sS + kQ_M * kQ_BN);
  float* sO = reinterpret_cast<float*>(sP + kQ_M * kQ_BN);
  float* sT = sO + kQ_M * kQ_D;
  float* sRowMax = sT + 16 * 256;
  float* sRowSum = sRowMax + kQ_M;
  float* sAlpha = sRowSum + kQ_M;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = blockIdx.x;

  const int safe_req = min(max(token_to_req[row], 0), nreq - 1);
  int valid = min(indices[(size_t)row * (TOPK + 1) + TOPK], TOPK);

  for (int i = tid; i < kQ_M * kQ_D; i += kQ_T) {
    const int m = i / kQ_D, d = i % kQ_D;
    sQ[i] = (m < G) ? q[((size_t)row * G + m) * kQ_D + d] : __float2half(0.f);
  }
  for (int i = tid; i < kQ_M * kQ_D; i += kQ_T) sO[i] = 0.0f;
  if (tid < kQ_M) {
    sRowMax[tid] = -1e30f;
    sRowSum[tid] = 0.0f;
    sAlpha[tid] = 1.0f;
  }
  __syncthreads();

  const int num_tiles = (valid + kQ_BN - 1) / kQ_BN;
  for (int t = 0; t < num_tiles; ++t) {
    const int base = t * kQ_BN;
    for (int i = tid; i < kQ_BN * kQ_D; i += kQ_T) {
      const int n = i / kQ_D, d = i % kQ_D;
      const int col = base + n;
      half kv = __float2half(0.f), vv = __float2half(0.f);
      if (col < valid) {
        const int tok = indices[(size_t)row * (TOPK + 1) + col];
        if (tok >= 0 && tok < page_width * kQ_PS) {
          const int phys = block_table[(size_t)safe_req * page_width
                                       + tok / kQ_PS];
          if (phys >= 0) {
            const size_t idx = ((size_t)phys * kQ_PS + tok % kQ_PS) * kQ_D + d;
            kv = k_cache[idx];
            vv = v_cache[idx];
          }
        }
      }
      sK[i] = kv;
      sV[i] = vv;
    }
    __syncthreads();

    if (warp * 16 < kQ_BN) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
      wmma::fill_fragment(acc, 0.0f);
#pragma unroll
      for (int k0 = 0; k0 < kQ_D; k0 += 16) {
        wmma::load_matrix_sync(af, sQ + k0, kQ_D);
        wmma::load_matrix_sync(bf, sK + warp * 16 * kQ_D + k0, kQ_D);
        wmma::mma_sync(acc, af, bf, acc);
      }
      for (int i = 0; i < acc.num_elements; ++i) acc.x[i] *= scale;
      wmma::store_matrix_sync(sS + warp * 16, acc, kQ_BN, wmma::mem_row_major);
    }
    __syncthreads();

    if (tid < kQ_M) {
      const int m = tid;
      float mx = -1e30f;
      for (int n = 0; n < kQ_BN; ++n)
        if (base + n < valid) mx = fmaxf(mx, sS[m * kQ_BN + n]);
      const float m_old = sRowMax[m];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (int n = 0; n < kQ_BN; ++n) {
        const float e = (base + n < valid) ? __expf(sS[m * kQ_BN + n] - m_new)
                                           : 0.0f;
        sP[m * kQ_BN + n] = __float2half(e);
        sum += e;
      }
      sAlpha[m] = (m_old < -1e29f) ? 0.0f : __expf(m_old - m_new);
      sRowSum[m] = sRowSum[m] * sAlpha[m] + sum;
      sRowMax[m] = m_new;
    }
    __syncthreads();

    for (int i = tid; i < kQ_M * kQ_D; i += kQ_T) sO[i] *= sAlpha[i / kQ_D];
    __syncthreads();

    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
#pragma unroll
      for (int dt = 0; (dt < 1) && (warp < 16); ++dt) {
        const int d0 = (warp * 1 + dt) * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int n0 = 0; n0 < kQ_BN; n0 += 16) {
          wmma::load_matrix_sync(pf, sP + n0, kQ_BN);
          wmma::load_matrix_sync(vf, sV + n0 * kQ_D + d0, kQ_D);
          wmma::mma_sync(acc, pf, vf, acc);
        }
        wmma::store_matrix_sync(sT + warp * 256, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
          const int m = i >> 4, d = i & 15;
          sO[m * kQ_D + d0 + d] += (sT + warp * 256)[i];
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < G * kQ_D; i += kQ_T) {
    const int m = i / kQ_D, d = i % kQ_D;
    const float inv = 1.0f / fmaxf(sRowSum[m], 1e-20f);
    const float g = __half2float(gate[((size_t)row * G + m) * kQ_D + d]);
    out[((size_t)row * G + m) * kQ_D + d] =
        __float2half(sO[m * kQ_D + d] * inv * qsigmoid_(g));
  }
}

constexpr int kQ_SmemBytes =
    (kQ_M * kQ_D + 2 * kQ_BN * kQ_D) * 2 + kQ_M * kQ_BN * 4
    + kQ_M * kQ_BN * 2 + kQ_M * kQ_D * 4 + 16 * 256 * 4 + 3 * kQ_M * 4;

// ---- T2.3b: FMA-layout kernel (M=4, row-major conflict-free smem) ----------
constexpr int kF_M = 4;    // GROUP_SIZE (3) padded to 4 (no WMMA -> tiny pad)
constexpr int kF_BN = 32;
constexpr int kF_D = 256;
constexpr int kF_T = 1024;

// 2nd arg = 2 CTAs/SM: pins ptxas at <=32 regs/thread (the baseline's
// exact register budget); a 47-reg version halves occupancy on V100
// (65536-reg file = 2048 thr x 32) and costs ~40%.
__global__ void __launch_bounds__(kF_T, 2)
qsa_prefill_fma_kernel(const half* __restrict__ q,
                       const half* __restrict__ k_cache,
                       const half* __restrict__ v_cache,
                       const int* __restrict__ indices,
                       const int* __restrict__ block_table,
                       const int* __restrict__ token_to_req,
                       half* __restrict__ out, const half* __restrict__ gate,
                       int G, int TOPK, int page_width, int nreq, float scale) {
  extern __shared__ __align__(16) unsigned char raw[];
  half* sQ = reinterpret_cast<half*>(raw);
  half* sK = sQ + kF_M * kF_D;
  half* sV = sK + kF_BN * kF_D;
  float* sS = reinterpret_cast<float*>(sV + kF_BN * kF_D);
  half* sP = reinterpret_cast<half*>(sS + kF_M * kF_BN);
  float* sO = reinterpret_cast<float*>(sP + kF_M * kF_BN);
  float* sRowMax = sO + kF_M * kF_D;
  float* sRowSum = sRowMax + kF_M;
  float* sAlpha = sRowSum + kF_M;
  int* sBaseA = reinterpret_cast<int*>(sAlpha + kF_M);
  int* sBaseB = sBaseA + kF_BN;  // T7: double sBase (prefetch next tile)

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int safe_req = min(max(token_to_req[row], 0), nreq - 1);
  int valid = min(indices[(size_t)row * (TOPK + 1) + TOPK], TOPK);

  // T7: sBase for tile t is computed at the END of tile t-1 (its global
  // gather overlaps tile t-1's PV work), so no per-tile barrier is spent
  // waiting for it. Tile 0 is resolved here in the prologue.
  auto load_base = [&](int tile, int* dst) {
    if (tid < kF_BN) {
      const int base = tile * kF_BN;
      int b = -1;
      if (base + tid < valid) {
        const int tok = indices[(size_t)row * (TOPK + 1) + base + tid];
        if (tok >= 0 && tok < page_width * kQ_PS) {
          const int phys =
              block_table[(size_t)safe_req * page_width + tok / kQ_PS];
          if (phys >= 0) b = (phys * kQ_PS + tok % kQ_PS) * kF_D;
        }
      }
      dst[tid] = b;
    }
  };

  // T7: resolve tile 0's sBase in the prologue so the prologue barrier
  // below also publishes it to all threads (tile 0's gather then has no
  // barrier of its own -- see the loop).
  load_base(0, sBaseA);
  for (int i = tid; i < kF_M * kF_D; i += kF_T) {
    const int m = i / kF_D, d = i % kF_D;
    sQ[i] = (m < G) ? q[((size_t)row * G + m) * kF_D + d] : __float2half(0.f);
  }
  for (int i = tid; i < kF_M * kF_D; i += kF_T) sO[i] = 0.0f;
  if (tid < kF_M) {
    sRowMax[tid] = -1e30f;
    sRowSum[tid] = 0.0f;
    sAlpha[tid] = 1.0f;
  }
  __syncthreads();

  const int num_tiles = (valid + kF_BN - 1) / kF_BN;
  int* sBase_cur = sBaseA;
  int* sBase_nxt = sBaseB;
  for (int t = 0; t < num_tiles; ++t) {
    const int base = t * kF_BN;
    // sBase_cur was resolved in tile t-1 (visible across its last barrier)
    // -- no barrier spent waiting for it here.
    for (int i = tid; i < kF_BN * kF_D / 2; i += kF_T) {
      const int n = (2 * i) / kF_D, d2 = (2 * i) % kF_D;
      const int b = sBase_cur[n];
      half2 kv = __float2half2_rn(0.f), vv = __float2half2_rn(0.f);
      if (b >= 0) {
        kv = *reinterpret_cast<const half2*>(k_cache + (size_t)b + d2);
        vv = *reinterpret_cast<const half2*>(v_cache + (size_t)b + d2);
      }
      reinterpret_cast<half2*>(sK)[i] = kv;
      reinterpret_cast<half2*>(sV)[i] = vv;
    }
    __syncthreads();

    // T7: issue the NEXT tile's sBase global loads now (two dependent
    // reads: indices -> block_table). They complete during scores/softmax/
    // PV below, so their ~1us latency is off this tile's critical path.
    int nxt_tok = -1, nxt_phys = -1;
    if (t + 1 < num_tiles && tid < kF_BN) {
      const int nb = (t + 1) * kF_BN;
      if (nb + tid < valid) {
        nxt_tok = indices[(size_t)row * (TOPK + 1) + nb + tid];
        if (nxt_tok >= 0 && nxt_tok < page_width * kQ_PS)
          nxt_phys = block_table[(size_t)safe_req * page_width + nxt_tok / kQ_PS];
      }
    }

    // scores: 1024 threads = 128 cells (m,n) x 8 d-sub; reduce over sub.
    {
      const int cell = tid >> 3;      // 0..127 = m*BN+n
      const int dsub = tid & 7;       // 0..7
      if (cell < kF_M * kF_BN) {
        const int m = cell / kF_BN, n = cell % kF_BN;
        const int col = base + n;
        float acc = 0.0f;
        if (col < valid) {
          const half2* q2 = reinterpret_cast<const half2*>(sQ + m * kF_D);
          const half2* k2 = reinterpret_cast<const half2*>(sK + n * kF_D);
          for (int d2 = dsub; d2 < kF_D / 2; d2 += 8) {
            const float2 qf = __half22float2(q2[d2]);
            const float2 kf = __half22float2(k2[d2]);
            acc = fmaf(qf.x, kf.x, acc);
            acc = fmaf(qf.y, kf.y, acc);
          }
        }
        // reduce the 8 d-subs (same warp, lanes are contiguous within a cell)
#pragma unroll
        for (int o = 4; o > 0; o >>= 1)
          acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (dsub == 0)
          sS[cell] = (col < valid) ? acc * scale : -1e30f;
      }
    }
    __syncthreads();

    if (tid < kF_M) {
      const int m = tid;
      float mx = -1e30f;
      for (int n = 0; n < kF_BN; ++n)
        if (base + n < valid) mx = fmaxf(mx, sS[m * kF_BN + n]);
      const float m_old = sRowMax[m];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (int n = 0; n < kF_BN; ++n) {
        const float e = (base + n < valid) ? __expf(sS[m * kF_BN + n] - m_new)
                                           : 0.0f;
        sP[m * kF_BN + n] = __float2half(e);
        sum += e;
      }
      sAlpha[m] = (m_old < -1e29f) ? 0.0f : __expf(m_old - m_new);
      sRowSum[m] = sRowSum[m] * sAlpha[m] + sum;
      sRowMax[m] = m_new;
    }
    __syncthreads();

    // T7: fused rescale+PV -- one thread owns sO[tid] for both ops
    // (kF_M*kF_D == kF_T, exactly one element per thread), so the old
    // barrier between the two is gone. The next tile's sBase is prefetched
    // here (threads < kF_BN) overlapping the PV; its write lands in
    // sBase_nxt and becomes visible at the tile-end barrier below.
    if (t + 1 < num_tiles && tid < kF_BN) {
      int b = -1;
      if (nxt_phys >= 0) b = (nxt_phys * kQ_PS + nxt_tok % kQ_PS) * kF_D;
      sBase_nxt[tid] = b;
    }
    if (tid < kF_M * kF_D) {
      const int m = tid / kF_D, d = tid % kF_D;
      float acc = sO[m * kF_D + d] * sAlpha[m];
#pragma unroll
      for (int n = 0; n < kF_BN; ++n)
        acc += __half2float(sP[m * kF_BN + n]) * __half2float(sV[n * kF_D + d]);
      sO[m * kF_D + d] = acc;
    }
    __syncthreads();
    // all threads see sBase_nxt here; swap before the next tile's gather
    int* tmp = sBase_cur;
    sBase_cur = sBase_nxt;
    sBase_nxt = tmp;
  }

  for (int i = tid; i < G * kF_D; i += kF_T) {
    const int m = i / kF_D, d = i % kF_D;
    const float inv = 1.0f / fmaxf(sRowSum[m], 1e-20f);
    const float g = __half2float(gate[((size_t)row * G + m) * kF_D + d]);
    out[((size_t)row * G + m) * kF_D + d] =
        __float2half(sO[m * kF_D + d] * inv * qsigmoid_(g));
  }
}

// ---- T10: phase-split copy of the FMA kernel (bench-only attribution) ----
// SKIP bitmask: 1 = zero-fill instead of the KV gather, 2 = skip the scores
// FMA (sS written as -1e30), 4 = skip the PV accumulation (the sBase store
// stays so the next tile's gather remains legal). Each skip is compile-time,
// so the variant costs what the full kernel costs minus that phase's
// arithmetic/memory (barriers and softmax always run, which is the
// attribution we want). Not on the serving path.
template <int SKIP>
__global__ void __launch_bounds__(kF_T, 2)
qsa_prefill_fma_phase_kernel(const half* __restrict__ q,
                             const half* __restrict__ k_cache,
                             const half* __restrict__ v_cache,
                             const int* __restrict__ indices,
                             const int* __restrict__ block_table,
                             const int* __restrict__ token_to_req,
                             half* __restrict__ out,
                             const half* __restrict__ gate,
                             int G, int TOPK, int page_width, int nreq,
                             float scale) {
  extern __shared__ __align__(16) unsigned char raw[];
  half* sQ = reinterpret_cast<half*>(raw);
  half* sK = sQ + kF_M * kF_D;
  half* sV = sK + kF_BN * kF_D;
  float* sS = reinterpret_cast<float*>(sV + kF_BN * kF_D);
  half* sP = reinterpret_cast<half*>(sS + kF_M * kF_BN);
  float* sO = reinterpret_cast<float*>(sP + kF_M * kF_BN);
  float* sRowMax = sO + kF_M * kF_D;
  float* sRowSum = sRowMax + kF_M;
  float* sAlpha = sRowSum + kF_M;
  int* sBaseA = reinterpret_cast<int*>(sAlpha + kF_M);
  int* sBaseB = sBaseA + kF_BN;

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int safe_req = min(max(token_to_req[row], 0), nreq - 1);
  int valid = min(indices[(size_t)row * (TOPK + 1) + TOPK], TOPK);

  auto load_base = [&](int tile, int* dst) {
    if (tid < kF_BN) {
      const int base = tile * kF_BN;
      int b = -1;
      if (base + tid < valid) {
        const int tok = indices[(size_t)row * (TOPK + 1) + base + tid];
        if (tok >= 0 && tok < page_width * kQ_PS) {
          const int phys =
              block_table[(size_t)safe_req * page_width + tok / kQ_PS];
          if (phys >= 0) b = (phys * kQ_PS + tok % kQ_PS) * kF_D;
        }
      }
      dst[tid] = b;
    }
  };

  load_base(0, sBaseA);
  for (int i = tid; i < kF_M * kF_D; i += kF_T) {
    const int m = i / kF_D, d = i % kF_D;
    sQ[i] = (m < G) ? q[((size_t)row * G + m) * kF_D + d] : __float2half(0.f);
  }
  for (int i = tid; i < kF_M * kF_D; i += kF_T) sO[i] = 0.0f;
  if (tid < kF_M) {
    sRowMax[tid] = -1e30f;
    sRowSum[tid] = 0.0f;
    sAlpha[tid] = 1.0f;
  }
  __syncthreads();

  const int num_tiles = (valid + kF_BN - 1) / kF_BN;
  int* sBase_cur = sBaseA;
  int* sBase_nxt = sBaseB;
  for (int t = 0; t < num_tiles; ++t) {
    const int base = t * kF_BN;
    for (int i = tid; i < kF_BN * kF_D / 2; i += kF_T) {
      const int n = (2 * i) / kF_D, d2 = (2 * i) % kF_D;
      const int b = sBase_cur[n];
      half2 kv = __float2half2_rn(0.f), vv = __float2half2_rn(0.f);
      if (!(SKIP & 1) && b >= 0) {
        kv = *reinterpret_cast<const half2*>(k_cache + (size_t)b + d2);
        vv = *reinterpret_cast<const half2*>(v_cache + (size_t)b + d2);
      }
      reinterpret_cast<half2*>(sK)[i] = kv;
      reinterpret_cast<half2*>(sV)[i] = vv;
    }
    __syncthreads();

    int nxt_tok = -1, nxt_phys = -1;
    if (t + 1 < num_tiles && tid < kF_BN) {
      const int nb = (t + 1) * kF_BN;
      if (nb + tid < valid) {
        nxt_tok = indices[(size_t)row * (TOPK + 1) + nb + tid];
        if (nxt_tok >= 0 && nxt_tok < page_width * kQ_PS)
          nxt_phys = block_table[(size_t)safe_req * page_width + nxt_tok / kQ_PS];
      }
    }

    {
      const int cell = tid >> 3;
      const int dsub = tid & 7;
      if (cell < kF_M * kF_BN) {
        const int m = cell / kF_BN, n = cell % kF_BN;
        const int col = base + n;
        float acc = 0.0f;
        if (!(SKIP & 2) && col < valid) {
          const half2* q2 = reinterpret_cast<const half2*>(sQ + m * kF_D);
          const half2* k2 = reinterpret_cast<const half2*>(sK + n * kF_D);
          for (int d2 = dsub; d2 < kF_D / 2; d2 += 8) {
            const float2 qf = __half22float2(q2[d2]);
            const float2 kf = __half22float2(k2[d2]);
            acc = fmaf(qf.x, kf.x, acc);
            acc = fmaf(qf.y, kf.y, acc);
          }
        }
#pragma unroll
        for (int o = 4; o > 0; o >>= 1)
          acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (dsub == 0)
          sS[cell] = (!(SKIP & 2) && col < valid) ? acc * scale : -1e30f;
      }
    }
    __syncthreads();

    if (tid < kF_M) {
      const int m = tid;
      float mx = -1e30f;
      for (int n = 0; n < kF_BN; ++n)
        if (base + n < valid) mx = fmaxf(mx, sS[m * kF_BN + n]);
      const float m_old = sRowMax[m];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.0f;
      for (int n = 0; n < kF_BN; ++n) {
        const float e = (base + n < valid) ? __expf(sS[m * kF_BN + n] - m_new)
                                           : 0.0f;
        sP[m * kF_BN + n] = __float2half(e);
        sum += e;
      }
      sAlpha[m] = (m_old < -1e29f) ? 0.0f : __expf(m_old - m_new);
      sRowSum[m] = sRowSum[m] * sAlpha[m] + sum;
      sRowMax[m] = m_new;
    }
    __syncthreads();

    if (t + 1 < num_tiles && tid < kF_BN) {
      int b = -1;
      if (nxt_phys >= 0) b = (nxt_phys * kQ_PS + nxt_tok % kQ_PS) * kF_D;
      sBase_nxt[tid] = b;
    }
    if (!(SKIP & 4) && tid < kF_M * kF_D) {
      const int m = tid / kF_D, d = tid % kF_D;
      float acc = sO[m * kF_D + d] * sAlpha[m];
#pragma unroll
      for (int n = 0; n < kF_BN; ++n)
        acc += __half2float(sP[m * kF_BN + n]) * __half2float(sV[n * kF_D + d]);
      sO[m * kF_D + d] = acc;
    }
    __syncthreads();
    int* tmp = sBase_cur;
    sBase_cur = sBase_nxt;
    sBase_nxt = tmp;
  }

  for (int i = tid; i < G * kF_D; i += kF_T) {
    const int m = i / kF_D, d = i % kF_D;
    const float inv = 1.0f / fmaxf(sRowSum[m], 1e-20f);
    const float g = __half2float(gate[((size_t)row * G + m) * kF_D + d]);
    out[((size_t)row * G + m) * kF_D + d] =
        __float2half(sO[m * kF_D + d] * inv * qsigmoid_(g));
  }
}

constexpr int kF_SmemBytes =
    (kF_M * kF_D + 2 * kF_BN * kF_D) * 2 + kF_M * kF_BN * 4
    + kF_M * kF_BN * 2 + kF_M * kF_D * 4 + 3 * kF_M * 4 + 2 * kF_BN * 4;

}  // namespace

torch::Tensor qsa_wmma_qk_probe(torch::Tensor q, torch::Tensor k, double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be fp16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(q.size(0) == kM && q.size(1) == kD, "q must be [16,256]");
  TORCH_CHECK(k.size(0) == kN && k.size(1) == kD, "k must be [64,256]");

  auto s = torch::empty({kM, kN}, q.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  wmma_qk_probe_kernel<<<1, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      s.data_ptr<float>(), static_cast<float>(scale));
  return s;
}

torch::Tensor qsa_wmma_attn_probe(torch::Tensor q, torch::Tensor k,
                                  torch::Tensor v, double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                  v.scalar_type() == at::kHalf,
              "inputs must be fp16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(q.size(0) == kM && q.size(1) == kD, "q must be [16,256]");
  TORCH_CHECK(k.size(0) == kN && k.size(1) == kD, "k must be [64,256]");
  TORCH_CHECK(v.size(0) == kN && v.size(1) == kD, "v must be [64,256]");

  auto o = torch::empty({kM, kD}, q.options());
  const int smem_bytes = (kA_M * kA_D + 2 * kA_N * kA_D) * 2
                         + kA_M * kA_N * 4 + kA_M * kA_N * 2 + 4 * 16 * 16 * 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(wmma_attn_probe_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    attr_set = true;
  }
  wmma_attn_probe_kernel<<<1, kThreads, smem_bytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<half*>(o.data_ptr<at::Half>()),
      static_cast<float>(scale));
  return o;
}

torch::Tensor qsa_prefill_sm70_forward(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor indices, torch::Tensor block_table,
    torch::Tensor token_to_req, torch::Tensor out, torch::Tensor gate,
    int64_t topk, double scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "q must be fp16 cuda");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && gate.is_contiguous(),
              "q/out/gate must be contiguous");
  TORCH_CHECK(indices.is_contiguous() && block_table.is_contiguous() &&
                  token_to_req.is_contiguous(),
              "index buffers must be contiguous");
  const int rows = static_cast<int>(q.size(0));
  const int G = static_cast<int>(q.size(1));
  const int page_width = static_cast<int>(block_table.size(1));
  const int nreq = static_cast<int>(block_table.size(0));
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(qsa_prefill_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kQ_SmemBytes);
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  qsa_prefill_attn_kernel<<<rows, kQ_T, kQ_SmemBytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
      indices.data_ptr<int>(), block_table.data_ptr<int>(),
      token_to_req.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()), G,
      static_cast<int>(topk), page_width, nreq, static_cast<float>(scale));
  return out;
}

torch::Tensor qsa_prefill_fma_forward(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor indices, torch::Tensor block_table,
    torch::Tensor token_to_req, torch::Tensor out, torch::Tensor gate,
    int64_t topk, double scale) {
  const int rows = static_cast<int>(q.size(0));
  const int G = static_cast<int>(q.size(1));
  const int page_width = static_cast<int>(block_table.size(1));
  const int nreq = static_cast<int>(block_table.size(0));
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(qsa_prefill_fma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kF_SmemBytes);
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  qsa_prefill_fma_kernel<<<rows, kF_T, kF_SmemBytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
      indices.data_ptr<int>(), block_table.data_ptr<int>(),
      token_to_req.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()), G,
      static_cast<int>(topk), page_width, nreq, static_cast<float>(scale));
  return out;
}

torch::Tensor qsa_prefill_fma_phase(torch::Tensor q, torch::Tensor k_cache,
                                     torch::Tensor v_cache,
                                     torch::Tensor indices,
                                     torch::Tensor block_table,
                                     torch::Tensor token_to_req,
                                     torch::Tensor out, torch::Tensor gate,
                                     int64_t topk, double scale,
                                     int64_t skip) {
  const int rows = static_cast<int>(q.size(0));
  const int G = static_cast<int>(q.size(1));
  const int page_width = static_cast<int>(block_table.size(1));
  const int nreq = static_cast<int>(block_table.size(0));
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(qsa_prefill_fma_phase_kernel<0>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kF_SmemBytes);
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* qp_ = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  const half* kc = reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>());
  const half* vc = reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>());
  const int* idx = indices.data_ptr<int>();
  const int* bt = block_table.data_ptr<int>();
  const int* t2r = token_to_req.data_ptr<int>();
  half* op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const half* gt = reinterpret_cast<const half*>(gate.data_ptr<at::Half>());
  const int tk = static_cast<int>(topk);
  const float sc = static_cast<float>(scale);
  switch (skip) {
    case 0:
      qsa_prefill_fma_phase_kernel<0><<<rows, kF_T, kF_SmemBytes, stream>>>(
          qp_, kc, vc, idx, bt, t2r, op, gt, G, tk, page_width, nreq, sc);
      break;
    case 1:
      qsa_prefill_fma_phase_kernel<1><<<rows, kF_T, kF_SmemBytes, stream>>>(
          qp_, kc, vc, idx, bt, t2r, op, gt, G, tk, page_width, nreq, sc);
      break;
    case 2:
      qsa_prefill_fma_phase_kernel<2><<<rows, kF_T, kF_SmemBytes, stream>>>(
          qp_, kc, vc, idx, bt, t2r, op, gt, G, tk, page_width, nreq, sc);
      break;
    case 4:
      qsa_prefill_fma_phase_kernel<4><<<rows, kF_T, kF_SmemBytes, stream>>>(
          qp_, kc, vc, idx, bt, t2r, op, gt, G, tk, page_width, nreq, sc);
      break;
    default:
      TORCH_CHECK(false, "skip must be 0/1/2/4");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qsa_wmma_qk_probe", &qsa_wmma_qk_probe, "T2.1 QK probe");
  m.def("qsa_wmma_attn_probe", &qsa_wmma_attn_probe, "T2.1b attn core");
  m.def("qsa_prefill_sm70_forward", &qsa_prefill_sm70_forward, "T2.2 WMMA");
  m.def("qsa_prefill_fma_forward", &qsa_prefill_fma_forward, "T2.3b FMA");
  m.def("qsa_prefill_fma_phase", &qsa_prefill_fma_phase,
        "T10 phase-split attribution (skip bitmask)");
}
