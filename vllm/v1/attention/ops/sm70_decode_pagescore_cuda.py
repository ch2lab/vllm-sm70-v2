"""CUDA page scorer for the SM70 nvfp4 decode cache (P3, engine-usable).

Scores every 16-token KV page against the current query without dequantizing the
whole cache: each page samples `S` token rows and, for every q head of that kv
head, keeps `max(sum_d q[head, d] * K[page, token, d])`.

Layout matches the decode kernels: page bytes are flat
`[K data H*N*DD][K scales H*N*SD][V data][V scales]`, `N=16`, `DD=D/2`,
`SD=D/16`. Only K is read, so one pass costs
`pages * S * G * D * 0.5 B` per kv head (score only; the attention itself then
runs on the selected pages).
"""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float e2m1_score(int n) {
  const float L[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = L[n & 7];
  return (n & 8) ? -v : v;
}

__device__ __forceinline__ float e4m3_score(unsigned char b) {
  int s = (b & 0x80) ? -1 : 1;
  int e = (b >> 3) & 0xF, m = b & 7;
  float v = (e == 0) ? (m * 0.001953125f) : ((1.f + m * 0.125f) * exp2f((float)(e - 7)));
  return s * v;
}

// grid (H, ntiles), block 32. Lane owns dims [lane*8, lane*8+8) = 4 packed bytes.
__global__ void page_scores_kernel(
    const __half* __restrict__ q,         // [HQ, D]
    const unsigned char* __restrict__ kv, // [B, ...] flat pages
    const int* __restrict__ bt,           // [bt_cols]
    const int* __restrict__ kvlens,       // [R]
    float* __restrict__ out,              // [H, ntiles]
    int HQ, int D, int NP, int DD, int SD, int page_bytes,
    int bt_cols, int ntiles, int samples, int r, int G) {
  const int h = blockIdx.x;
  const int pg = blockIdx.y;
  const int lane = threadIdx.x;
  const int kv_len = kvlens[r];
  float best = -1e30f;
  if (pg * NP < kv_len) {
    const int bid = bt[r * bt_cols + pg];
    const unsigned char* base = kv + (long)bid * page_bytes;
    const unsigned char* kdata = base + (long)h * (NP * DD);
    const unsigned char* kscale =
        base + (long)gridDim.x * (NP * DD) + (long)h * (NP * SD);
    for (int s = 0; s < samples; ++s) {
      const int tok = (int)((long)s * NP / samples);
      if (pg * NP + tok >= kv_len) break;
      const unsigned char* kd = kdata + (long)tok * DD + lane * 4;
      const float sc = e4m3_score(kscale[(long)tok * SD + (lane >> 1)]);
      const unsigned int w = *reinterpret_cast<const unsigned int*>(kd);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int qh = h * G + g;
        if (qh >= HQ) break;
        const __half* qp = q + (long)qh * D + lane * 8;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned char byte = (w >> (j * 8)) & 0xFFu;
          acc += __half2float(qp[j * 2]) * e2m1_score(byte & 0xF) * sc;
          acc += __half2float(qp[j * 2 + 1]) * e2m1_score((byte >> 4) & 0xF) * sc;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_xor_sync(0xffffffffu, acc, off);
        best = fmaxf(best, acc);
      }
    }
  }
  if (lane == 0) out[h * ntiles + pg] = best;
}

void page_scores(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
                 torch::Tensor kvlens, torch::Tensor out, int64_t samples,
                 int64_t r) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = out.size(0), ntiles = out.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  dim3 grid(H, ntiles);
  page_scores_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      out.data_ptr<float>(), HQ, D, NP, DD, SD, page_bytes,
      (bt.dim() == 2) ? (int)bt.size(1) : (int)bt.size(0), ntiles,
      (int)samples, (int)r, G);
}


// Coalesced page-mean K: grid (H, ceil(pages/PP)), block 256 (one thread per dim).
// Reads each page's K rows sequentially (128 B/token, coalesced) and writes the
// page mean, so scoring becomes one small GEMV per head instead of a gather per
// page. PP pages per CTA keeps the launch count small.
__global__ void page_mean_k_kernel(
    const unsigned char* __restrict__ kv, const int* __restrict__ bt,
    const int* __restrict__ kvlens, __half* __restrict__ kmean,
    int D, int NP, int DD, int SD, int page_bytes, int bt_cols, int ntiles,
    int r, int PP) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;               // 0..255, one dim
  const int sub = d >> 4;                  // 16-value scale group
  const int kv_len = kvlens[r];
  for (int pg = blockIdx.y; pg < ntiles; pg += gridDim.y) {
    if (pg * NP >= kv_len) break;
    const int bid = bt[r * bt_cols + pg];
    const unsigned char* kdata = kv + (long)bid * page_bytes + (long)h * (NP * DD);
    const unsigned char* kscale = kv + (long)bid * page_bytes +
                                  (long)gridDim.x * (NP * DD) +
                                  (long)h * (NP * SD);
    float acc = 0.f;
    int n = 0;
    for (int t = 0; t < NP; ++t) {
      if (pg * NP + t >= kv_len) break;
      const unsigned char byte = kdata[(long)t * DD + (d >> 1)];
      const int code = ((d & 1) ? (byte >> 4) : (byte & 0xF));
      const float sc = e4m3_score(kscale[(long)t * SD + sub]);
      acc += e2m1_score(code) * sc;
      ++n;
    }
    kmean[((long)h * ntiles + pg) * D + d] = __float2half(acc / (float)(n ? n : 1));
  }
}

void page_mean_k(torch::Tensor kv, torch::Tensor bt, torch::Tensor kvlens,
                 torch::Tensor kmean, int64_t r, int64_t pp) {
  const int H = kmean.size(0), ntiles = kmean.size(1), D = kmean.size(2);
  const int NP = kv.size(2), F = kv.size(3);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  dim3 grid(H, (ntiles + (int)pp - 1) / (int)pp);
  page_mean_k_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      reinterpret_cast<__half*>(kmean.data_ptr<at::Half>()), D, NP, DD, SD,
      page_bytes, (bt.dim() == 2) ? (int)bt.size(1) : (int)bt.size(0),
      ntiles, (int)r, (int)pp);
}


// Coarse scorer: one CTA per (kv head, superblock of SUPER tokens), 2 sampled
// token rows, 128 threads (2 dims each), block reduce. Reads ~256 B per CTA,
// so the whole pass is O(nsuper) and negligible next to the attention it gates.
__global__ void super_scores_kernel(
    const __half* __restrict__ q, const unsigned char* __restrict__ kv,
    const int* __restrict__ bt, const int* __restrict__ kvlens,
    float* __restrict__ out, int HQ, int D, int NP, int DD, int SD,
    int page_bytes, int bt_cols, int nsuper, int ntok, int r, int G,
    int SUPER, int samples) {
  const int h = blockIdx.x, sb = blockIdx.y;
  const int tid = threadIdx.x;               // 0..127 -> dims [2*tid, 2*tid+2)
  __shared__ float red[128];
  // the first token of the superblock decides which page it lives in
  const int tok0 = sb * SUPER;
  const int kv_len = kvlens[r];
  float best = -1e30f;
  if (tok0 < kv_len) {
    for (int s = 0; s < samples; ++s) {
      const int tok = tok0 + (int)((long)s * SUPER / samples);
      if (tok >= kv_len) break;
      const int pg = tok / NP, tt = tok % NP;
      const int bid = bt[r * bt_cols + pg];
      const unsigned char* base = kv + (long)bid * page_bytes;
      const unsigned char* kdata = base + (long)h * (NP * DD) + (long)tt * DD;
      const unsigned char* ksc = base + (long)gridDim.x * (NP * DD) +
                                 (long)h * (NP * SD) + (long)tt * SD;
      const int d = tid * 2;
      const unsigned char byte = kdata[d >> 1];
      const float sc = e4m3_score(ksc[d >> 4]);
      const float k0 = e2m1_score(byte & 0xF) * sc;
      const float k1 = e2m1_score((byte >> 4) & 0xF) * sc;
      for (int g = 0; g < G; ++g) {
        const int qh = h * G + g;
        if (qh >= HQ) break;
        const __half* qp = q + (long)qh * D + d;
        float acc = __half2float(qp[0]) * k0 + __half2float(qp[1]) * k1;
        red[tid] = acc;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
          if (tid < st) red[tid] += red[tid + st];
          __syncthreads();
        }
        best = fmaxf(best, red[0]);
        __syncthreads();
      }
    }
  }
  if (tid == 0) out[h * nsuper + sb] = best;
}

void super_scores(torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
                  torch::Tensor kvlens, torch::Tensor out, int64_t r,
                  int64_t super_, int64_t samples) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = out.size(0), nsuper = out.size(1);
  const int NP = kv.size(2), F = kv.size(3);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  dim3 grid(H, nsuper);
  super_scores_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt.data_ptr<int>(), kvlens.data_ptr<int>(),
      out.data_ptr<float>(), HQ, D, NP, DD, SD, page_bytes,
      (bt.dim() == 2) ? (int)bt.size(1) : (int)bt.size(0), nsuper, 0,
      (int)r, G, (int)super_, (int)samples);
}


// One CTA per kv head: exact top-k by pairwise rank (nsuper <= 1088), then the
// compacted table: [top-k superblocks' pages][sink pages][recent window][tail].
__global__ void build_table_kernel(const float* __restrict__ scores,
                                   int nsuper, int per_sb,
                                   const int* __restrict__ bt_row, int cols,
                                   const int* __restrict__ kv_lens, int npg,
                                   int n_pick_sb, int sink, int win,
                                   int* __restrict__ sel_bt,
                                   int* __restrict__ sel_len,
                                   int* __restrict__ state, int dense_max) {
  if (state && state[3] == 0) return;
  __shared__ float sm[1088];
  const int tid = threadIdx.x;
  const int nh = gridDim.y;
  const int kv_len = kv_lens[0];
  const int npages = (kv_len + npg - 1) / npg;
  // The budget is a *maximum*: on a short context there may be fewer pages than
  // the budget, and writing the full budget anyway (with pages clamped to the
  // last one) made sel_len exceed kv_len -- the kernel then attended to the same
  // tail page many times over, costing 6.4x the work at 10K and skewing the
  // softmax.  Cap the budget at what the request actually has.
  const int max_sb = (npages + per_sb - 1) / per_sb;
  const int n_pick = min(n_pick_sb, max_sb);
  int* out = sel_bt;
  /* Below `dense_max` pages the selector cannot pay for itself (attention is
     already cheap), so stay bit-for-bit dense: identity table, no ranking. */
  if (n_pick >= max_sb || (dense_max > 0 && npages <= dense_max)) {
    // The budget covers the whole context: there is nothing to select.  Emit the
    // identity page table so this step is bit-for-bit the dense path (same page
    // order, same length) instead of a re-ranked permutation of everything --
    // a permutation cannot be faster, and it made short contexts behave
    // differently for no gain.
    for (int i = tid; i < npages; i += blockDim.x)
      out[i] = bt_row[min(i, cols - 1)];
    __syncthreads();
    if (tid == 0) {
      sel_len[0] = kv_len;
      if (state) { state[4] = 0; state[5] = 1; }   /* identity: tail is a no-op */
    }
    return;
  }
  for (int i = tid; i < nsuper; i += blockDim.x) {
    float s = -1e30f;
    for (int h = 0; h < nh; ++h)
      s = fmaxf(s, scores[(long)h * nsuper + i]);
    sm[i] = s;
  }
  __syncthreads();
  for (int i = tid; i < nsuper; i += blockDim.x) {
    const float s = sm[i];
    if (s <= -1e29f) continue;
    int rank = 0;
    for (int j = 0; j < nsuper; ++j)
      if (sm[j] > s || (sm[j] == s && j < i)) ++rank;
    if (rank < n_pick) {
      for (int j = 0; j < per_sb; ++j) {
        const int pg = min(i * per_sb + j, npages - 1);
        out[rank * per_sb + j] = bt_row[min(pg, cols - 1)];
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    int pos = n_pick * per_sb;
    for (int i = 0; i < sink; ++i) out[pos++] = bt_row[min(i, cols - 1)];
    const int w0 = max(npages - win, sink);
    for (int i = 0; i < win; ++i) out[pos++] = bt_row[min(w0 + i, cols - 1)];
    out[pos++] = bt_row[min(max(npages - 1, 0), cols - 1)];
    const int tail = kv_len - (npages - 1) * npg;
    const int slen = (pos - 1) * npg + tail;
    sel_len[0] = slen < kv_len ? slen : kv_len;
  }
  if (state) {
    state[4] = n_pick * per_sb + sink;   /* where the tail region starts */
    state[5] = 0;
  }
}

/* Periodic re-selection gate.  state[] = [calls, last_sel_call, last_kv,
   active, tail_off, identity, 0, 0].  Under CUDA-graph capture every layer
   replays the same host code, so the decision has to live on the device:
   the table is rebuilt only once every `period_calls` selector calls, and
   immediately when the request got shorter (a new request). */
__global__ void sel_gate_kernel(const int* __restrict__ kvlens,
                                int* __restrict__ state, int period_calls) {
  const int ctr = state[0] + 1;
  state[0] = ctr;
  const int kv = kvlens[0];
  const bool active = (period_calls <= 1) || (state[1] == 0)
                      || (ctr - state[1] >= period_calls) || (kv < state[2]);
  if (active) { state[1] = ctr; state[2] = kv; }
  state[3] = active ? 1 : 0;
}

/* The tail of the compacted table (recent window + last page) is rewritten on
   every call, so tokens written since the last selection stay visible even when
   the ranking itself is reused.  Runs after build_table (or alone when the gate
   is closed) and only ever touches the tail region. */
__global__ void sel_tail_kernel(const int* __restrict__ bt, int cols, int npg,
                                const int* __restrict__ kvlens, int sink,
                                int win, int* __restrict__ state,
                                int* __restrict__ sel_bt,
                                int* __restrict__ sel_len) {
  if (blockIdx.x != 0) return;
  const int kv_len = kvlens[0];
  const int npages = (kv_len + npg - 1) / npg;
  if (state[5] != 0) {
    /* identity mode: the whole table is the request's own page list, so it has
       to grow with the context (one int copy per page, cheap). */
    for (int i = threadIdx.x; i < npages; i += blockDim.x)
      sel_bt[i] = bt[min(i, cols - 1)];
    if (threadIdx.x == 0) sel_len[0] = kv_len;
    return;
  }
  if (threadIdx.x != 0) return;
  int pos = state[4];
  const int w0 = max(npages - win, sink);
  for (int i = w0; i < npages; ++i) sel_bt[pos++] = bt[min(i, cols - 1)];
  if (pos <= state[4]) pos = state[4] + 1;
  const int tail = kv_len - (npages - 1) * npg;
  const int slen = (pos - 1) * npg + tail;
  sel_len[0] = slen < kv_len ? slen : kv_len;
}

void select_pages(torch::Tensor q, torch::Tensor kv, torch::Tensor bt_row,
                  torch::Tensor kv_lens, torch::Tensor sel_bt,
                  torch::Tensor sel_len, int64_t npg, int64_t super_,
                  int64_t samples, int64_t n_pick_pages, int64_t sink,
                  int64_t win) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = sel_bt.size(0) ? kv.size(1) / 2 : kv.size(1) / 2;
  const int NP = kv.size(2);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  const int cols = (int)bt_row.numel();
  const int nsuper = (int)(((long)cols * NP + super_ - 1) / super_);
  const int per_sb = (int)(super_ / NP);
  const int n_pick_sb = (int)std::max<int64_t>(1, n_pick_pages / per_sb);
  auto scores = torch::empty({H, nsuper},
                             q.options().dtype(torch::kFloat));
  dim3 g1(H, nsuper);
  super_scores_kernel<<<g1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      kv.data_ptr<unsigned char>(), bt_row.data_ptr<int>(),
      kv_lens.data_ptr<int>(), scores.data_ptr<float>(), HQ, D, NP, DD, SD,
      page_bytes, cols, nsuper, 0, 0, G, (int)super_, (int)samples);
  dim3 g2(1, H);
  build_table_kernel<<<g2, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), nsuper, per_sb, bt_row.data_ptr<int>(), cols,
      kv_lens.data_ptr<int>(), (int)npg, n_pick_sb, (int)sink, (int)win,
      sel_bt.data_ptr<int>(), sel_len.data_ptr<int>(), nullptr, 0);
}


// Mean mode: exact page-mean K over ALL tokens of every page, then one GEMV per
// kv head. No sampling, so a short needle cannot be skipped by the sampler; the
// cost is one linear pass over K (0.8 ms at 250K), amortised across layers.
void select_pages_mean(torch::Tensor q, torch::Tensor kv, torch::Tensor bt_row,
                       torch::Tensor kv_lens, torch::Tensor sel_bt,
                       torch::Tensor sel_len, torch::Tensor kmean,
                       int64_t npg, int64_t super_, int64_t n_pick_pages,
                       int64_t sink, int64_t win) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = kv.size(1) / 2;
  const int NP = kv.size(2);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  const int cols = (int)bt_row.numel();
  const int nsuper = (int)((cols + super_ / npg - 1) / (super_ / npg));
  const int per_sb = (int)(super_ / npg);
  const int npages_max = cols;
  auto opts = q.options();
  // 1) page-mean K over every token: [H, cols, D]
  dim3 g1(H, (cols + 15) / 16);
  page_mean_k_kernel<<<g1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<unsigned char>(), bt_row.data_ptr<int>(),
      kv_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(kmean.data_ptr<at::Half>()), D, NP, DD, SD,
      page_bytes, cols, cols, 0, 16);
  // 2) score = mean of the superblock's page means, per (kv head, superblock)
  auto qf = q.to(torch::kFloat);                       // [HQ, D]
  auto km = kmean.to(torch::kFloat);                   // [H, cols, D]
  auto page_sc = at::bmm(km, qf.view({H, G, D}).mean(1).unsqueeze(-1))
                     .squeeze(-1);                     // [H, cols]
  // average the per-page scores inside each superblock
  auto pad = per_sb - (cols % per_sb);
  auto sc = page_sc;
  if (pad != per_sb) sc = at::constant_pad_nd(sc, {0, pad}, 0);
  auto scores = sc.view({H, -1, per_sb}).mean(2);       // [H, nsuper]
  // 3) same rank/table kernel as the sampled path
  dim3 g2(1, H);
  build_table_kernel<<<g2, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), nsuper, per_sb, bt_row.data_ptr<int>(), cols,
      kv_lens.data_ptr<int>(), (int)npg, (int)std::max<int64_t>(1, n_pick_pages / per_sb),
      (int)sink, (int)win, sel_bt.data_ptr<int>(), sel_len.data_ptr<int>(),
      nullptr, 0);
}

// Gather the prefill Quest prior into superblock scores: every page of a
// superblock carries the same value, so a max over `per_sb` pages is enough.
// Superblocks at or past kv_len stay -inf so they can never enter the top-k.
__global__ void prior_scores_kernel(const float* __restrict__ prior,
                                    const int* __restrict__ bt_row,
                                    const int* __restrict__ kv_lens,
                                    float* __restrict__ scores, int nsuper,
                                    int per_sb, int cols, int npg) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nsuper) return;
  const int kv_len = kv_lens[0];
  float v = -1e30f;
  if ((long)i * per_sb * npg < kv_len) {
    for (int j = 0; j < per_sb; ++j) {
      const int pg = min(i * per_sb + j, cols - 1);
      if ((long)pg * npg >= kv_len) break;
      v = fmaxf(v, prior[bt_row[pg]]);
    }
  }
  scores[i] = v;
}

void select_pages_prior(torch::Tensor prior, torch::Tensor bt_row,
                        torch::Tensor kv_lens, torch::Tensor sel_bt,
                        torch::Tensor sel_len, int64_t npg, int64_t super_,
                        int64_t n_pick_pages, int64_t sink, int64_t win) {
  const int cols = (int)bt_row.numel();
  const int per_sb = (int)(super_ / npg);
  const int nsuper = (int)((cols + per_sb - 1) / per_sb);
  auto scores = torch::empty({1, nsuper}, prior.options());
  prior_scores_kernel<<<(nsuper + 255) / 256, 256, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      prior.data_ptr<float>(), bt_row.data_ptr<int>(), kv_lens.data_ptr<int>(),
      scores.data_ptr<float>(), nsuper, per_sb, cols, (int)npg);
  const int n_pick_sb = (int)std::max<int64_t>(1, n_pick_pages / per_sb);
  dim3 g2(1, 1);
  build_table_kernel<<<g2, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), nsuper, per_sb, bt_row.data_ptr<int>(), cols,
      kv_lens.data_ptr<int>(), (int)npg, n_pick_sb, (int)sink, (int)win,
      sel_bt.data_ptr<int>(), sel_len.data_ptr<int>(), nullptr, 0);
}

// Exact Quest upper-bound metadata: per-page K min/max over all tokens of the
// page (dequantized).  Selection then uses max(q*kmax, q*kmin) per dim, so a
// single high-scoring token (e.g. a needle) cannot be lost to averaging or
// sampling.
__global__ void page_minmax_k_kernel(
    const unsigned char* __restrict__ kv, const int* __restrict__ bt,
    const int* __restrict__ kvlens, __half* __restrict__ kmin,
    __half* __restrict__ kmax, int D, int NP, int DD, int SD,
    int page_bytes, int bt_cols, int ntiles, int r, int PP,
    const int* __restrict__ state, int kstride) {
  if (state[3] == 0) return;   /* periodic re-selection: reuse the last table */
  const int h = blockIdx.x;
  const int p0 = blockIdx.y * PP;
  const int d = threadIdx.x;
  const int sub = d >> 4;
  const int kv_len = kvlens[r];
  // 256-entry LUTs: the scalar e4m3/e2m1 conversion was the whole 2.4 ms/layer
  // (PP=16 and PP=1 measured identical), so replace it with one smem read.
  __shared__ float s_e2[16];
  __shared__ float s_e4[256];
  if (d < 16) s_e2[d] = e2m1_score(d);
  s_e4[d] = e4m3_score((unsigned char)d);
  __syncthreads();
  for (int pg = p0; pg < p0 + PP && pg < ntiles; ++pg) {
    if (pg * NP >= kv_len) break;
    const int bid = bt[r * bt_cols + pg];
    const unsigned char* kdata =
        kv + (long)bid * page_bytes + (long)h * (NP * DD);
    const unsigned char* kscale = kv + (long)bid * page_bytes +
                                  (long)gridDim.x * (NP * DD) +
                                  (long)h * (NP * SD);
    float lo = 1e30f, hi = -1e30f;
    int n = 0;
    for (int t = 0; t < NP; ++t) {
      if (pg * NP + t >= kv_len) break;
      const unsigned char byte = kdata[(long)t * DD + (d >> 1)];
      const int code = ((d & 1) ? (byte >> 4) : (byte & 0xF));
      const float sc = s_e4[kscale[(long)t * SD + sub]];
      const float v = s_e2[code] * sc;
      lo = fminf(lo, v);
      hi = fmaxf(hi, v);
      ++n;
    }
    kmin[((long)h * kstride + pg) * D + d] = __float2half(n ? lo : 0.f);
    kmax[((long)h * kstride + pg) * D + d] = __float2half(n ? hi : 0.f);
  }
}

// score(h, page) = sum_d max over q-head of max(q*kmax, q*kmin)
__global__ void page_bound_score_kernel(
    const __half* __restrict__ q, const __half* __restrict__ kmin,
    const __half* __restrict__ kmax, float* __restrict__ out,
    int HQ, int H, int D, int cols, const int* __restrict__ state,
    int kstride, int npg, const int* __restrict__ kvlens) {
  if (state[3] == 0) return;
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  const int G = HQ / H;
  const int kv_len = kvlens[0];
  __shared__ float red[256];
  for (int pg = blockIdx.y; pg < cols; pg += gridDim.y) {
    if (pg * npg >= kv_len) break;
    const float lo = __half2float(kmin[((long)h * kstride + pg) * D + d]);
    const float hi = __half2float(kmax[((long)h * kstride + pg) * D + d]);
    float best = -1e30f;
    for (int g = 0; g < G; ++g) {
      const float qv = __half2float(q[((long)(h * G + g)) * D + d]);
      best = fmaxf(best, fmaxf(qv * hi, qv * lo));
    }
    red[d] = best;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (d < s) red[d] += red[d + s];
      __syncthreads();
    }
    if (d == 0) out[h * cols + pg] = red[0];
    __syncthreads();
  }
}

__global__ void page_to_super_max_kernel(
    const float* __restrict__ pg, float* __restrict__ out, int cols,
    int per_sb, int nsuper, int H, const int* __restrict__ state) {
  if (state[3] == 0) return;
  const int h = blockIdx.x, sb = blockIdx.y;
  const int tid = threadIdx.x;
  const int base = sb * per_sb;
  float m = -1e30f;
  for (int i = tid; i < per_sb && base + i < cols; i += blockDim.x)
    m = fmaxf(m, pg[h * cols + base + i]);
  __shared__ float red[256];
  red[tid] = m;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out[h * nsuper + sb] = red[0];
}


// Exact page score: max over the page's 16 tokens and the G query heads of
// q.k_t.  This is the true top logit any token of the page can contribute, so
// ranking by it cannot lose a needle to the looseness of a min/max bound (the
// bound sums |q|*max|k| per dim, which ranks by K norm rather than relevance).
// Thread layout mirrors the XQA KV reader: 16 tokens x 16 scale groups, each
// thread dequantising 16 fp4 dims (uint2) with one e4m3 scale.
__global__ void page_exact_score_kernel(
    const unsigned char* __restrict__ kv, const int* __restrict__ bt,
    const int* __restrict__ kvlens, const __half* __restrict__ q,
    float* __restrict__ out, int HQ, int H, int D, int NP, int DD, int SD,
    int page_bytes, int cols, int G, int r, const int* __restrict__ state) {
  if (state[3] == 0) return;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int stok = tid >> 4, ssub = tid & 15;
  const int kv_len = kvlens[r];
  __shared__ float s_e2[16];
  __shared__ float s_e4[256];
  __shared__ __half sq[8][256];
  __shared__ float sdot[16][8];
  if (tid < 16) s_e2[tid] = e2m1_score(tid);
  s_e4[tid] = e4m3_score((unsigned char)tid);
  for (int g = 0; g < G; ++g)
    if (tid < D) sq[g][tid] = q[((long)(h * G + g)) * D + tid];
  __syncthreads();
  for (int pg = blockIdx.y; pg < cols; pg += gridDim.y) {
    if (pg * NP >= kv_len) break;
    const int bid = bt[r * cols + pg];
    const unsigned char* base = kv + (long)bid * page_bytes;
    const bool live = (pg * NP + stok) < kv_len;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = 0.f;
    if (live) {
      const unsigned char* kd = base + (long)h * (NP * DD)
                                + (long)stok * DD + ssub * 8;
      const unsigned char* ks = base + (long)H * (NP * DD)
                                + (long)h * (NP * SD) + (long)stok * SD + ssub;
      const uint2 raw = *reinterpret_cast<const uint2*>(kd);
      const float sc = s_e4[*ks];
      const unsigned int w[2] = {raw.x, raw.y};
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int code = (w[j] >> (b * 8)) & 0xFFu;
          v[j * 8 + b * 2] = s_e2[code & 0xF] * sc;
          v[j * 8 + b * 2 + 1] = s_e2[code >> 4] * sc;
        }
    }
    float best = -1e30f;
    for (int g = 0; g < G; ++g) {
      float a = 0.f;
      if (live) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
          a += v[i] * __half2float(sq[g][ssub * 16 + i]);
      } else {
        a = -1e30f;
      }
      for (int off = 8; off > 0; off >>= 1)
        a += __shfl_down_sync(0xffffffffu, a, off, 16);
      if (ssub == 0) sdot[stok][g] = a;
    }
    __syncthreads();
    if (tid == 0) {
      /* One writer: the score is the max over BOTH tokens and query heads, so
         several threads must not each store their own head's value. */
      float m = -1e30f;
#pragma unroll
      for (int t = 0; t < 16; ++t)
        for (int g = 0; g < G; ++g) m = fmaxf(m, sdot[t][g]);
      out[h * cols + pg] = m;
    }
    __syncthreads();
  }
}

/* Host code must never dereference a device pointer: force the gate open with a
   kernel, exactly like the periodic gate does. */
__global__ void sel_gate_force_kernel(int* __restrict__ state) { state[3] = 1; }

void page_exact_scores(torch::Tensor q, torch::Tensor kv, torch::Tensor bt_row,
                       torch::Tensor kv_lens, torch::Tensor out,
                       torch::Tensor state, int64_t r) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = kv.size(1) / 2;
  const int NP = kv.size(2);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int G = HQ / H;
  const int cols = (int)bt_row.numel();
  int* st = state.data_ptr<int>();
  sel_gate_force_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(st);
  const int gcols = (int)std::min<long>(cols, 512L);
  page_exact_score_kernel<<<dim3(H, gcols), 256, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      kv.data_ptr<unsigned char>(), bt_row.data_ptr<int>(),
      kv_lens.data_ptr<int>(),
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      out.data_ptr<float>(), HQ, H, D, NP, DD, SD, page_bytes, cols, G,
      (int)r, st);
}

void select_pages_minmax(torch::Tensor q, torch::Tensor kv,
                         torch::Tensor bt_row, torch::Tensor kv_lens,
                         torch::Tensor sel_bt, torch::Tensor sel_len,
                         torch::Tensor kmin, torch::Tensor kmax,
                         int64_t npg, int64_t super_, int64_t n_pick_pages,
                         int64_t sink, int64_t win, int64_t scan,
                         torch::Tensor state, int64_t period_calls,
                         int64_t dense_max, int64_t exact) {
  const int HQ = q.size(0), D = q.size(1);
  const int H = kv.size(1) / 2;
  const int NP = kv.size(2);
  const int DD = D / 2, SD = D / 16;
  const int page_bytes = 2 * (H * NP * DD + H * NP * SD);
  const int cols = (int)bt_row.numel();
  const int per_sb = (int)(super_ / npg);
  const int nsuper = (int)((cols + per_sb - 1) / per_sb);
  auto opts = q.options();
  int* st = state.data_ptr<int>();
  /* kmin/kmax are allocated wide (262144 tokens plus slack) and reused for any
     block-table width, so their row stride is not `cols`.  Getting this wrong
     silently loaded the wrong page's bound. */
  const int kstride = (int)kmin.size(1);
  const int G = HQ / H;
  sel_gate_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      kv_lens.data_ptr<int>(), st, (int)period_calls);
  /* Cap the page-parallel grids: with 16848 block-table columns the launch
     itself cost ~0.9 ms/step at 10K, where every block early-exits. */
  const int gcols = (int)std::min<long>(cols, 512L);
  auto psc = torch::empty({H, cols}, opts.dtype(torch::kFloat));
  if (exact) {
    /* One pass over K per page, scored exactly: no min/max metadata needed. */
    page_exact_score_kernel<<<dim3(H, gcols), 256, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        kv.data_ptr<unsigned char>(), bt_row.data_ptr<int>(),
        kv_lens.data_ptr<int>(),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        psc.data_ptr<float>(), HQ, H, D, NP, DD, SD, page_bytes, cols, G, 0,
        st);
  } else {
    dim3 g1(H, gcols);
    if (scan)
    page_minmax_k_kernel<<<g1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        kv.data_ptr<unsigned char>(), bt_row.data_ptr<int>(),
        kv_lens.data_ptr<int>(),
        reinterpret_cast<__half*>(kmin.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(kmax.data_ptr<at::Half>()), D, NP, DD, SD,
        page_bytes, cols, cols, 0, 1, st, kstride);
    dim3 g2(H, gcols);
    page_bound_score_kernel<<<g2, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kmin.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kmax.data_ptr<at::Half>()),
        psc.data_ptr<float>(), HQ, H, D, cols, st, kstride, (int)npg,
        kv_lens.data_ptr<int>());
  }
  auto ssc = torch::empty({H, nsuper}, opts.dtype(torch::kFloat));
  dim3 g3(H, nsuper);
  page_to_super_max_kernel<<<g3, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      psc.data_ptr<float>(), ssc.data_ptr<float>(), cols, per_sb, nsuper, H,
      st);
  dim3 g4(1, H);
  build_table_kernel<<<g4, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      ssc.data_ptr<float>(), nsuper, per_sb, bt_row.data_ptr<int>(), cols,
      kv_lens.data_ptr<int>(), (int)npg,
      (int)std::max<int64_t>(1, n_pick_pages / per_sb), (int)sink, (int)win,
      sel_bt.data_ptr<int>(), sel_len.data_ptr<int>(), st, (int)dense_max);
  sel_tail_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      bt_row.data_ptr<int>(), cols, (int)npg, kv_lens.data_ptr<int>(), (int)sink,
      (int)win, st, sel_bt.data_ptr<int>(), sel_len.data_ptr<int>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("page_scores", &page_scores);
  m.def("page_mean_k", &page_mean_k);
  m.def("super_scores", &super_scores);
  m.def("select_pages", &select_pages);
  m.def("select_pages_mean", &select_pages_mean);
  m.def("select_pages_prior", &select_pages_prior);
  m.def("select_pages_minmax", &select_pages_minmax);
  m.def("page_exact_scores", &page_exact_scores);
}
"""

_MOD = None
_STATE0 = None

# Page-keyed Quest prior written by the block-sparse prefill (see
# sm70_nvfp4_prefill_twopass.run_tiles_sparse).  Pages of one 256-token
# superblock share a score, so the decode selector only has to gather it.
_PRIOR = None
_PRIOR_N = 0


def prior_buffer(num_pages: int, device):
    """Persistent [num_pages] fp32 prior; created once per cache geometry."""
    global _PRIOR, _PRIOR_N
    if _PRIOR is None or _PRIOR.numel() < num_pages or _PRIOR.device != device:
        _PRIOR = torch.full(
            (num_pages,), -1e30, dtype=torch.float32, device=device
        )
    _PRIOR_N = num_pages
    return _PRIOR


def prior_ready() -> bool:
    return _PRIOR is not None and _PRIOR_N > 0


def select_pages_prior(bt_row, kvlens, sel_bt, sel_len, npg=16, super_=256,
                       n_pick_pages=2048, sink=0, win=0):
    """Rank pages by the prefill Quest prior instead of a sampled estimate."""
    _mod().select_pages_prior(_PRIOR, bt_row, kvlens, sel_bt, sel_len, npg,
                              super_, n_pick_pages, sink, win)
    return sel_bt, sel_len


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(name="sm70_decode_pagescore", cpp_sources="",
                           cuda_sources=_CUDA, verbose=False,
                           extra_cuda_cflags=["-O3", "--use_fast_math"])
    return _MOD


def page_scores(q, kv, bt, kvlens, out, samples=4, r=0):
    _mod().page_scores(q, kv, bt, kvlens, out, samples, r)
    return out


def super_scores(q, kv, bt, kvlens, out, r=0, super_=256, samples=2):
    """Coarse per-(kv head, superblock) score; `out` is [H, nsuper] fp32."""
    _mod().super_scores(q, kv, bt, kvlens, out, r, super_, samples)
    return out


def select_pages(q, kv, bt_row, kvlens, sel_bt, sel_len, npg=16, super_=256,
                 samples=2, n_pick_pages=2048, sink=0, win=0):
    """One call: score superblocks, exact top-k, build the compacted table."""
    _mod().select_pages(q, kv, bt_row, kvlens, sel_bt, sel_len, npg, super_,
                        samples, n_pick_pages, sink, win)
    return sel_bt, sel_len


def select_pages_minmax(q, kv, bt_row, kvlens, sel_bt, sel_len, kmin, kmax,
                        npg=16, super_=256, n_pick_pages=2048, sink=0, win=0,
                        scan=True, state=None, period_calls=1,
                        dense_max=0, exact=False):
    """Exact Quest bound: per-page K min/max, score = max(q*kmax, q*kmin).

    ``scan=False`` reuses the min/max metadata across layers of the same step;
    only the per-layer score/table pass runs.

    ``period_calls`` > 1 rebuilds the page table only once every that many
    selector calls (device-side gate, CUDA-graph safe) and refreshes just the
    recent window in between.
    """
    global _STATE0
    if state is None:
        # A gate that is always open, so single-call users keep the old
        # behaviour without threading a buffer through.
        if _STATE0 is None or _STATE0.device != q.device:
            _STATE0 = torch.zeros(8, dtype=torch.int32, device=q.device)
        state = _STATE0
    _mod().select_pages_minmax(q, kv, bt_row, kvlens, sel_bt, sel_len, kmin,
                               kmax, npg, super_, n_pick_pages, sink, win,
                               int(scan), state, int(period_calls),
                               int(dense_max), int(exact))
    return sel_bt, sel_len


def select_pages_mean(q, kv, bt_row, kvlens, sel_bt, sel_len, kmean, npg=16,
                      super_=256, n_pick_pages=2048, sink=0, win=0):
    """Page-mean selector: exact mean over all tokens of every page."""
    _mod().select_pages_mean(q, kv, bt_row, kvlens, sel_bt, sel_len, kmean,
                             npg, super_, n_pick_pages, sink, win)
    return sel_bt, sel_len


def page_mean_k(kv, bt, kvlens, kmean, r=0, pp=16):
    """Page-mean K, coalesced: [H, pages, D] fp16 in `kmean`."""
    _mod().page_mean_k(kv, bt, kvlens, kmean, r, pp)
    return kmean

def page_exact_scores(q, kv, bt_row, kvlens, out, state, r=0):
    """Per-(head, page) exact max_t(q.k_t); `out` is [H, cols] fp32."""
    _mod().page_exact_scores(q, kv, bt_row, kvlens, out, state, r)
    return out
