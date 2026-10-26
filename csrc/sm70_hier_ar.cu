// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// ============================================================================
// SM70 (V100) topology-aware hierarchical all-reduce — round-3 AR shadow
// route (decode/verify-size only). New-tree-native implementation; NOTHING
// is copied from another tree (rule 11671 — idea-level sources only):
//
//   Idea source (recorded per house rule 11671):
//   /data/src/1cat-vllm/docs/design/sm70_deepseek_v4_tp8_graph_allreduce.md
//     * hierarchical 2-clique protocol: 4-way intra-clique reduce forming an
//       FP32 clique partial in fixed global-rank order, mirror-pair
//       cross-clique exchange over the single cross-quad NVLink leg, one
//       final FP32 add + single FP16 downcast (bitwise-equal across ranks);
//     * double-buffered graph-safe handshakes — clique-ready, pair-ready and
//       the partial/data slots are all double-buffered, with the local-input
//       consumption acknowledgement published CONCURRENTLY with the pair
//       exchange (the 1Cat accepted protocol; its rejected paths — one
//       signal/partial slot, partial-only double buffering, serialized
//       clique completion, deferred fence polling — are not built here);
//     * device-side waits replacing the collective graph-launch
//       coordination (the graph-scheduling win);
//     * the sidecar-library override env idea (VLLM_SM70_CUSTOM_AR_LIBRARY)
//       -> VLLM_SM70_AR_LIB (bench-only seam).
//   House patterns (this tree): csrc/libtorch_stable/custom_all_reduce.cu
//   (allocate_shared_buffer_and_handle / open_mem_handle IPC exchange),
//   csrc/custom_collective_common.cuh (release/acquire flag idioms for SM70:
//   st.release.sys / ld.acquire.sys), custom_all_reduce.py
//   create_shared_buffer (IPC handle allgather).
//
//   Contract: .superpowers/sdd/2026-10-13-sm70-round3-ar/ar-direction-spec.md
//   (topology §1.1, protocol §2.a, integration §3, gate-1 §4).
//
// ============================================================================
// Topology (boot-verified by the Python route; fail-closed on mismatch):
//   quads  A = {0,1,2,3}   B = {4,5,6,7}  — each a full 4-GPU NVLink clique
//   mirror legs (the ONLY cross-quad NVLink legs; zero SYS legs by
//   construction): 0<->7 (NV1), 1<->6 (NV1), 2<->5 (NV2), 3<->4 (NV2)
//
// Per-rank device buffer (one cudaMalloc + one IPC handle, allocated at
// boot before any graph capture — kernel args are therefore fixed at
// capture time and graph-safe by construction; no deferred-pointer
// registration is needed because no graph-pool buffer is involved):
//
//   [0, 64)                      meta: 8 x u32
//   [64, 64 + 4*n_max)           in_buf: 2 slots x n_max x fp16
//   [64 + 4*n_max,
//    64 + 16*n_max)               region: 2 slots x 6*n_max bytes, each slot:
//                                    staging @0 (n_max x fp32), payload @4n
//                                    (wire mode: e4m3 n_max B + scales @5n
//                                    fp32 n_max/128; or fp16 wire, 2*n_max B
//                                    spanning [4n..6n), no scales)
//   [64 + 16*n_max, ... + 8)     probe: int32 id, int32 ok (kProbeWords=2)
//
// Meta word map (u32):
//   [0] epoch            — device-side counter, +1 per kernel launch; slot =
//                           epoch & 1 (alternating slots, per 1Cat protocol)
//   [1..2] in_ready[s]    — epoch at which my input for slot s is staged
//   [3..4] consumed[s]    — epoch at which my 4 clique inputs for slot s have
//                           been consumed (local-input consumption ack)
//   [5..6] partial_ready[s] — epoch at which my clique partial/wire for slot s
//                           is ready at the mirror
//
// ============================================================================
// Protocol (1 CTA, 512 threads; single kernel per all-reduce, graph-safe):
//
//   E = ++meta.epoch; s = E & 1; prev = E - 2
//   1. [if E >= 2] wait for the 3 clique peers' consumed[s] == prev
//      (slot-reuse gate: my slot s is re-staged only after every clique
//      reader has acknowledged consumption of it two epochs ago — the 1Cat
//      concurrent local-consumption acknowledgement; bounds clique overtake
//      to exactly one collective, which the alternating slot absorbs).
//   2. stage the local input into my in_buf slot s; CTA-wide system
//      release; publish in_ready[s] = E.
//   3. wait (3 concurrent acquires) for the 3 clique peers' in_ready[s]
//      == E.
//   4. form the FP32 clique partial: acc = in[q0] + in[q1] + in[q2] + in[q3]
//      with q0..q3 = the quad's global ranks in ascending order — every rank
//      reads the same four inputs in the same order (rank-replicated
//      accumulation), so the partial is bitwise identical across the quad.
//   5. write the partial (or its e4m3 wire encoding) to my slot; publish
//      consumed[s] = E (clique ack) and partial_ready[s] = E (mirror
//      signal) CONCURRENTLY (two release stores, no serialization between
//      them and no wait on the other side's completion).
//   6. wait for the mirror's partial_ready[s] == E; read the mirror's
//      partial (dequantized in wire mode).
//   7. total = partial(quad 0) + partial(quad 1) in that fixed global
//      order (identical expression on all 8 ranks); single FP16 downcast.
//
// Overtake analysis (why double-buffered slots + the ack are sufficient):
//   * input slot: re-staged only after the 3 clique consumption acks of
//     E-2 (step 1) — direct gate, independent of any other timing.
//   * partial slot / partial_ready: rank r rewrites its slot s at E+2 only
//     after r's E+1 completes; r's E+1 waits on the mirror's
//     partial_ready[E+1], which the mirror publishes during ITS E+1 —
//     strictly after the mirror finished ITS E (the only place it read
//     r's slot s). Hence the mirror has always consumed r's slot s before
//     r's E+2 write; the 1-collective mirror overtake is absorbed by the
//     alternating slot. (This is exactly the protocol class the 1Cat
//     8,700-collective graph stress validated; no extra mirror ack is
//     invented.)
//   * signals are double-buffered so a stale epoch can never be
//     overwritten by a newer one before its reader passes the wait (the
//     1Cat rejected-path "signal overtake" is the failure this prevents).
// ============================================================================

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "torch_utils.h"

namespace {

using fptr_t = int64_t;

#define SM70_HIER_AR_CUDA_CHECK(cmd)                                      \
  do {                                                                    \
    cudaError_t e_ = (cmd);                                               \
    if (e_ != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("sm70_hier_ar: ") + #cmd +     \
                               " failed: " + cudaGetErrorString(e_));      \
    }                                                                     \
  } while (0)

constexpr int kThreads = 512;  // one CTA (1Cat's accepted protocol uses one
                               // 512-thread CTA; idea-level only)
constexpr size_t kMetaBytes = 64;  // 8 x u32
constexpr int kProbeWords = 2;  // int32 id, int32 ok

// Flag idioms (SM70: PTX release/acquire sys scope — same as
// csrc/custom_collective_common.cuh for __CUDA_ARCH__ >= 700).
__device__ __forceinline__ void st_flag_release(uint32_t* addr, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(addr));
}
__device__ __forceinline__ uint32_t ld_flag_acquire(const uint32_t* addr) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
  return v;
}

// ----------------------------------------------------------------------------
// Wire codec (VLLM_SM70_AR_FP8_WIRE=1; INFORMATIONAL ONLY — spec §2.b verdict:
// dead weight on the current wiring, carried for the post-rewire re-tune and
// bench sweeps). Per-128-element group scale (fp32), payload e4m3 (software
// codec — SM70 has no HW fp8 convert). Every rank quantizes an identical
// partial (rank-replicated) and every rank dequantizes the same payload bits
// with the same scale, so the result stays bitwise-equal across all 8 ranks
// (spec precision gate).
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint8_t f32_to_e4m3(float x) {
  uint8_t sign = x < 0.f ? 0x80u : 0x00u;
  float ax = x < 0.f ? -x : x;
  if (ax >= 448.0f) return sign | 0x7Eu;  // saturate to the e4m3fn max (448)
  if (ax <= 0.0f) return sign;
  int e2;
  float m = frexpf(ax, &e2);  // ax = m * 2^e2, m in [0.5, 1)
  if (e2 - 1 <= -7) {  // k = floor(log2 ax) <= -7: subnormal (f/8)*2^-6
    uint32_t f = (uint32_t)__float2int_rn(ax * 512.0f);  // round-to-nearest-even
    if (f > 7u) {
      if (f == 8u) return sign | 0x08u;  // rounds to 1.0*2^-6 (E=1, f=0)
      f = 7u;
    }
    return sign | (uint8_t)f;
  }
  int E = e2 + 6;  // exponent field (bias 7); k = e2-1 >= -6 -> E >= 1
  // ax * 2^(7-E) in [1, 2): ax = m*2^e2, so == m * 2 (EXACT — m in [0.5,1),
  // one power-of-2 scale keeps the 24-bit significand intact).
  // (BUG 2026-10-15: this line was `ax * __int2float_rn(128 - e2)` — an
  // INTEGER multiply by ~126, not 2^(7-E): every value carried to the
  // next binade (f=0), silently dropping the 3-bit e4m3 mantissa.)
  float scaled = m * 2.0f;
  uint32_t f = (uint32_t)__float2int_rn((scaled - 1.0f) * 8.0f);
  if (f > 7u) {
    f = 0;
    E += 1;  // mantissa carry
  }
  if (E >= 15) {
    // E=15,f=7 is the e4m3fn NaN slot; unreachable here (ax < 448 is
    // pre-clamped) but clamp anyway.
    if (E == 15 && f <= 6u) return sign | (uint8_t)(((uint32_t)E << 3) | f);
    return sign | 0x7Eu;
  }
  return sign | (uint8_t)(((uint32_t)E << 3) | f);
}

// e4m3 decode: 256-entry LUT in constant memory — BIT-EXACT BY CONSTRUCTION
// (hex-float literals of the exact grid values; sign pre-baked per byte).
// One LDC per decode, no arithmetic, no branches — and immune to the
// -O3 context-dependent miscompilation of the inlined arithmetic decoder
// (observed 2026-10-15: identical source compiled two ways with only
// extra stores differing rounded one boundary sum differently; the
// arithmetic decoder's 1.0f+f/8.0f * ldexpf(...) context-flipped).
// Table generated 2026-10-15 from the spec: E=0 -> (f/8)*2^-6, else
// (1+f/8)*2^(E-7); byte b = sign | E<<3 | f.
__device__ __constant__ float c_e4m3_lut[256] = {
    0x0.0p+0f,
0x1.0000000000000p-9f,
    0x1.0000000000000p-8f,
    0x1.8000000000000p-8f,
    0x1.0000000000000p-7f,
    0x1.4000000000000p-7f,
    0x1.8000000000000p-7f,
    0x1.c000000000000p-7f,
    0x1.0000000000000p-6f,
    0x1.2000000000000p-6f,
    0x1.4000000000000p-6f,
    0x1.6000000000000p-6f,
    0x1.8000000000000p-6f,
    0x1.a000000000000p-6f,
    0x1.c000000000000p-6f,
    0x1.e000000000000p-6f,
    0x1.0000000000000p-5f,
    0x1.2000000000000p-5f,
    0x1.4000000000000p-5f,
    0x1.6000000000000p-5f,
    0x1.8000000000000p-5f,
    0x1.a000000000000p-5f,
    0x1.c000000000000p-5f,
    0x1.e000000000000p-5f,
    0x1.0000000000000p-4f,
    0x1.2000000000000p-4f,
    0x1.4000000000000p-4f,
    0x1.6000000000000p-4f,
    0x1.8000000000000p-4f,
    0x1.a000000000000p-4f,
    0x1.c000000000000p-4f,
    0x1.e000000000000p-4f,
    0x1.0000000000000p-3f,
    0x1.2000000000000p-3f,
    0x1.4000000000000p-3f,
    0x1.6000000000000p-3f,
    0x1.8000000000000p-3f,
    0x1.a000000000000p-3f,
    0x1.c000000000000p-3f,
    0x1.e000000000000p-3f,
    0x1.0000000000000p-2f,
    0x1.2000000000000p-2f,
    0x1.4000000000000p-2f,
    0x1.6000000000000p-2f,
    0x1.8000000000000p-2f,
    0x1.a000000000000p-2f,
    0x1.c000000000000p-2f,
    0x1.e000000000000p-2f,
    0x1.0000000000000p-1f,
    0x1.2000000000000p-1f,
    0x1.4000000000000p-1f,
    0x1.6000000000000p-1f,
    0x1.8000000000000p-1f,
    0x1.a000000000000p-1f,
    0x1.c000000000000p-1f,
    0x1.e000000000000p-1f,
    0x1.0000000000000p+0f,
    0x1.2000000000000p+0f,
    0x1.4000000000000p+0f,
    0x1.6000000000000p+0f,
    0x1.8000000000000p+0f,
    0x1.a000000000000p+0f,
    0x1.c000000000000p+0f,
    0x1.e000000000000p+0f,
    0x1.0000000000000p+1f,
    0x1.2000000000000p+1f,
    0x1.4000000000000p+1f,
    0x1.6000000000000p+1f,
    0x1.8000000000000p+1f,
    0x1.a000000000000p+1f,
    0x1.c000000000000p+1f,
    0x1.e000000000000p+1f,
    0x1.0000000000000p+2f,
    0x1.2000000000000p+2f,
    0x1.4000000000000p+2f,
    0x1.6000000000000p+2f,
    0x1.8000000000000p+2f,
    0x1.a000000000000p+2f,
    0x1.c000000000000p+2f,
    0x1.e000000000000p+2f,
    0x1.0000000000000p+3f,
    0x1.2000000000000p+3f,
    0x1.4000000000000p+3f,
    0x1.6000000000000p+3f,
    0x1.8000000000000p+3f,
    0x1.a000000000000p+3f,
    0x1.c000000000000p+3f,
    0x1.e000000000000p+3f,
    0x1.0000000000000p+4f,
    0x1.2000000000000p+4f,
    0x1.4000000000000p+4f,
    0x1.6000000000000p+4f,
    0x1.8000000000000p+4f,
    0x1.a000000000000p+4f,
    0x1.c000000000000p+4f,
    0x1.e000000000000p+4f,
    0x1.0000000000000p+5f,
    0x1.2000000000000p+5f,
    0x1.4000000000000p+5f,
    0x1.6000000000000p+5f,
    0x1.8000000000000p+5f,
    0x1.a000000000000p+5f,
    0x1.c000000000000p+5f,
    0x1.e000000000000p+5f,
    0x1.0000000000000p+6f,
    0x1.2000000000000p+6f,
    0x1.4000000000000p+6f,
    0x1.6000000000000p+6f,
    0x1.8000000000000p+6f,
    0x1.a000000000000p+6f,
    0x1.c000000000000p+6f,
    0x1.e000000000000p+6f,
    0x1.0000000000000p+7f,
    0x1.2000000000000p+7f,
    0x1.4000000000000p+7f,
    0x1.6000000000000p+7f,
    0x1.8000000000000p+7f,
    0x1.a000000000000p+7f,
    0x1.c000000000000p+7f,
    0x1.e000000000000p+7f,
    0x1.0000000000000p+8f,
    0x1.2000000000000p+8f,
    0x1.4000000000000p+8f,
    0x1.6000000000000p+8f,
    0x1.8000000000000p+8f,
    0x1.a000000000000p+8f,
    0x1.c000000000000p+8f,
    0x1.e000000000000p+8f,
    -0x0.0p+0f,
    -0x1.0000000000000p-9f,
    -0x1.0000000000000p-8f,
    -0x1.8000000000000p-8f,
    -0x1.0000000000000p-7f,
    -0x1.4000000000000p-7f,
    -0x1.8000000000000p-7f,
    -0x1.c000000000000p-7f,
    -0x1.0000000000000p-6f,
    -0x1.2000000000000p-6f,
    -0x1.4000000000000p-6f,
    -0x1.6000000000000p-6f,
    -0x1.8000000000000p-6f,
    -0x1.a000000000000p-6f,
    -0x1.c000000000000p-6f,
    -0x1.e000000000000p-6f,
    -0x1.0000000000000p-5f,
    -0x1.2000000000000p-5f,
    -0x1.4000000000000p-5f,
    -0x1.6000000000000p-5f,
    -0x1.8000000000000p-5f,
    -0x1.a000000000000p-5f,
    -0x1.c000000000000p-5f,
    -0x1.e000000000000p-5f,
    -0x1.0000000000000p-4f,
    -0x1.2000000000000p-4f,
    -0x1.4000000000000p-4f,
    -0x1.6000000000000p-4f,
    -0x1.8000000000000p-4f,
    -0x1.a000000000000p-4f,
    -0x1.c000000000000p-4f,
    -0x1.e000000000000p-4f,
    -0x1.0000000000000p-3f,
    -0x1.2000000000000p-3f,
    -0x1.4000000000000p-3f,
    -0x1.6000000000000p-3f,
    -0x1.8000000000000p-3f,
    -0x1.a000000000000p-3f,
    -0x1.c000000000000p-3f,
    -0x1.e000000000000p-3f,
    -0x1.0000000000000p-2f,
    -0x1.2000000000000p-2f,
    -0x1.4000000000000p-2f,
    -0x1.6000000000000p-2f,
    -0x1.8000000000000p-2f,
    -0x1.a000000000000p-2f,
    -0x1.c000000000000p-2f,
    -0x1.e000000000000p-2f,
    -0x1.0000000000000p-1f,
    -0x1.2000000000000p-1f,
    -0x1.4000000000000p-1f,
    -0x1.6000000000000p-1f,
    -0x1.8000000000000p-1f,
    -0x1.a000000000000p-1f,
    -0x1.c000000000000p-1f,
    -0x1.e000000000000p-1f,
    -0x1.0000000000000p+0f,
    -0x1.2000000000000p+0f,
    -0x1.4000000000000p+0f,
    -0x1.6000000000000p+0f,
    -0x1.8000000000000p+0f,
    -0x1.a000000000000p+0f,
    -0x1.c000000000000p+0f,
    -0x1.e000000000000p+0f,
    -0x1.0000000000000p+1f,
    -0x1.2000000000000p+1f,
    -0x1.4000000000000p+1f,
    -0x1.6000000000000p+1f,
    -0x1.8000000000000p+1f,
    -0x1.a000000000000p+1f,
    -0x1.c000000000000p+1f,
    -0x1.e000000000000p+1f,
    -0x1.0000000000000p+2f,
    -0x1.2000000000000p+2f,
    -0x1.4000000000000p+2f,
    -0x1.6000000000000p+2f,
    -0x1.8000000000000p+2f,
    -0x1.a000000000000p+2f,
    -0x1.c000000000000p+2f,
    -0x1.e000000000000p+2f,
    -0x1.0000000000000p+3f,
    -0x1.2000000000000p+3f,
    -0x1.4000000000000p+3f,
    -0x1.6000000000000p+3f,
    -0x1.8000000000000p+3f,
    -0x1.a000000000000p+3f,
    -0x1.c000000000000p+3f,
    -0x1.e000000000000p+3f,
    -0x1.0000000000000p+4f,
    -0x1.2000000000000p+4f,
    -0x1.4000000000000p+4f,
    -0x1.6000000000000p+4f,
    -0x1.8000000000000p+4f,
    -0x1.a000000000000p+4f,
    -0x1.c000000000000p+4f,
    -0x1.e000000000000p+4f,
    -0x1.0000000000000p+5f,
    -0x1.2000000000000p+5f,
    -0x1.4000000000000p+5f,
    -0x1.6000000000000p+5f,
    -0x1.8000000000000p+5f,
    -0x1.a000000000000p+5f,
    -0x1.c000000000000p+5f,
    -0x1.e000000000000p+5f,
    -0x1.0000000000000p+6f,
    -0x1.2000000000000p+6f,
    -0x1.4000000000000p+6f,
    -0x1.6000000000000p+6f,
    -0x1.8000000000000p+6f,
    -0x1.a000000000000p+6f,
    -0x1.c000000000000p+6f,
    -0x1.e000000000000p+6f,
    -0x1.0000000000000p+7f,
    -0x1.2000000000000p+7f,
    -0x1.4000000000000p+7f,
    -0x1.6000000000000p+7f,
    -0x1.8000000000000p+7f,
    -0x1.a000000000000p+7f,
    -0x1.c000000000000p+7f,
    -0x1.e000000000000p+7f,
    -0x1.0000000000000p+8f,
    -0x1.2000000000000p+8f,
    -0x1.4000000000000p+8f,
    -0x1.6000000000000p+8f,
    -0x1.8000000000000p+8f,
    -0x1.a000000000000p+8f,
    -0x1.c000000000000p+8f,
    -0x1.e000000000000p+8f
};

__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
  // One LDC; bit-exact by construction (header above).
  return c_e4m3_lut[b];
}


// NVFP4 wire codec (OCP FP4), ported from ch2lab/vllm sm120-nvfp4
// b57e19d1e5 (e2m1 payload, 2 codes/byte even-in-low-nibble, per-16 E4M3
// scale s = e4m3(amax/6)). Pure function of the input -> bitwise-equal
// across ranks (same cross-rank guarantee as the e4m3 path). Error bound
// <= 0.25 * block amax (e2m1 0.167 + scale rounding 0.0625).
constexpr int NV4_BLK = 16;  // elements per E4M3 scale
__device__ __forceinline__ uint32_t nv4_e2m1_code(float v) {
  // v = |x| / s in [0, ~6]; RTNE on the OCP grid {0,.5,1,1.5,2,3,4,6}
  // (odd-mantissa bins: >= on the upper tie; even: >) — mirrors the
  // reference bit-for-bit.
  uint32_t c = (v > 0.25f) + (v >= 0.75f) + (v > 1.25f) + (v >= 1.75f) +
               (v > 2.5f) + (v >= 3.5f) + (v > 5.0f);
  return c & 7u;
}
__device__ __constant__ float c_nv4_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

__device__ __forceinline__ float nv4_e2m1_val(uint32_t c) {
  // 8-entry LUT, same rationale as c_e4m3_lut (bit-exact by
  // construction, one LDC, no arithmetic context to miscompile).
  return c_nv4_lut[c & 7u];
}

// Per-call kernel argument block (fixed at graph-capture time — all pointers
// are boot-time IPC bases; only n/wire vary per call).
// Phase instrumentation (VLLM_SM70_AR_PHASE=1): %globaltimer stamps at the
// protocol boundaries so the per-call cost can be split into stage / clique
// wait / quad reduce+publish / mirror wait. Disabled => args.phase == nullptr
// and each stamp is one predictable branch.
// Always-on AR telemetry: per wire mode {calls, total_ns, total_bytes, max_ns}.
// Four atomics from one thread per collective (~0.2 us of a ~22 us call); this
// is the "real load" signal for analysis and for any future adaptive policy.
__device__ unsigned long long g_sm70_ar_stats[4 * 4];
// Per-hop breakdown (hop 0 = quad reduce incl. staging, hop 1 = mirror leg +
// combine). The three quad peers move identical bytes over the same window, so
// the useful split is hop-level, not peer-level.
__device__ unsigned long long g_sm70_ar_hop[2 * 4];

#define SM70_AR_PHASE(a, i)                                                  \
  do {                                                                       \
    if ((a).phase != nullptr && threadIdx.x == 0) {                          \
      unsigned long long sm70_t_;                                            \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(sm70_t_));            \
      (a).phase[i] = sm70_t_;                                                \
    }                                                                        \
  } while (0)

struct SM70HierARArgs {
  uint32_t* meta;          // own meta
  half* in_buf;            // own in slots (base)
  uint8_t* region;         // own region slots (base)
  const half* quad_in_buf[4];      // quad members' in_buf bases, rank order
  const uint32_t* peer_in_ready[3];  // 3 clique peers' meta+1 (in_ready[0])
  const uint32_t* peer_consumed[3];  // 3 clique peers' meta+3 (consumed[0])
  const uint8_t* mirror_region;  // mirror's region base
  const uint32_t* mirror_partial_ready;  // mirror's meta+5
  int rank;
  int own_quad;  // 0 or 1
  int n_max;  // slot stride in elements (n%128==0 for all calls)
  int n;  // elements this call
  int wire;  // 0 = fp32 partial, 1 = e4m3, 2 = nvfp4, 3 = fp16
  bool stats;                 // per-wire counters on (default on)
  unsigned long long* stats_buf;  // host-mapped counters [24] or null
};

__global__ void __launch_bounds__(kThreads, 1)
sm70_hier_ar_kernel(SM70HierARArgs a, const half* __restrict__ input,
                    half* __restrict__ out) {
  __shared__ uint32_t s_epoch;
  const int tid = threadIdx.x;
  const int T = kThreads;
  if (tid == 0) {
    s_epoch = atomicAdd(&a.meta[0], 1u);  // stream-serialized: single CTA
  }
  __syncthreads();
  const uint32_t E = s_epoch;
  // Decide sampling before touching the timer: 7 of 8 collectives then pay
  // nothing at all (the two %globaltimer reads alone cost ~0.05 ms/step).
  const bool ar_sample = a.stats && (a.stats_buf != nullptr) && ((E & 7u) == 0);
  unsigned long long t_entry = 0;
  if (ar_sample) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_entry));
  }
  const int s = E & 1;
  const uint32_t prev = (E >= 2) ? E - 2 : 0;
  // in_buf = 2 slots x n_max x fp16 (buf_size / header L46): slot s sits at
  // s * n_max halves. (BUG 2026-10-15: was 2 * n_max — s=1 then landed at
  // in_buf + 4*n_max bytes = region_+0, so every s=1 epoch's step-2 input
  // write clobbered the first n/2 fp32 of MY OWN slot-0 staging; writer
  // and reader agreed on the same wrong offset so s=1 epochs 'worked',
  // but slot-0 staging was silently trampled. Unit m=64 w1 dump: the
  // stale half matched the previous s=1 epoch's input 100%.)
  const size_t in_stride = (size_t)a.n_max;               // in slots, elems
  const size_t region_stride = 6 * (size_t)a.n_max;       // region slots, bytes
  const size_t payload_off = 4 * (size_t)a.n_max;
  const size_t scales_off = 5 * (size_t)a.n_max;

  half* own_in = a.in_buf + s * in_stride;
  uint8_t* own_region = a.region + s * region_stride;
  float* staging = (float*)own_region;
  const half* quad_slot[4];
#pragma unroll
  for (int k = 0; k < 4; k++) quad_slot[k] = a.quad_in_buf[k] + s * in_stride;

  // (1) slot-reuse gate: all 3 clique readers must have consumed my slot s
  //     two epochs ago (their consumed[s] == prev).
  if (E >= 2) {
    if (tid < 3) {
      while (ld_flag_acquire(a.peer_consumed[tid] + s) != prev) {
      }
    }
    __syncthreads();
  }

  // (2) stage the local input into my slot (peer-visible via IPC).
  for (int i = tid; i < a.n; i += T) own_in[i] = input[i];
  // CTA-wide system release: every thread fences its own stores, the CTA
  // barrier orders all fences before the flag.
  __syncthreads();
  // Device scope is sufficient here: every payload byte travels GPU-to-GPU
  // over NVLink into a peer's device memory (no host/pinned-memory path), and
  // the flag publish/consume pair is already system-scope
  // (st.release.sys / ld.acquire.sys). This is the NCCL convention for
  // intra-node device-to-device handshakes; __threadfence_system() ordered
  // host-visible memory we never touch and cost ~1.7 us per call (2 fences x
  // 98 calls/step = 0.34 ms, measured 78.0 -> 80.1 tok/s decode1).
  // Revert to __threadfence_system() if this kernel ever publishes to host
  // memory or crosses nodes.
  __threadfence();
  __syncthreads();
  if (tid == 0) st_flag_release(a.meta + 1 + s, E);

  // (3) wait for the 3 clique peers' inputs (3 concurrent acquires).
  if (tid < 3) {
    while (ld_flag_acquire(a.peer_in_ready[tid] + s) != E) {
    }
  }
  __syncthreads();

  // (4) FP32 clique partial in fixed global-rank order. quad_in_buf[] is in
  // ascending global-rank order, so this accumulation is the fixed order on
  // every rank (rank-replicated -> bitwise-identical partial in the quad).
  // wire=3 goes straight to the half payload: the fp32 staging write and the
  // separate conversion pass (previously two full passes) collapse into one.
  half* pay16 = (a.wire == 3) ? (half*)(own_region + payload_off) : nullptr;
  if (pay16) {
    for (int i = 2 * tid; i < a.n; i += 2 * T) {
      float a0 = __half2float(quad_slot[0][i]);
      a0 += __half2float(quad_slot[1][i]);
      a0 += __half2float(quad_slot[2][i]);
      a0 += __half2float(quad_slot[3][i]);
      float a1 = __half2float(quad_slot[0][i + 1]);
      a1 += __half2float(quad_slot[1][i + 1]);
      a1 += __half2float(quad_slot[2][i + 1]);
      a1 += __half2float(quad_slot[3][i + 1]);
      *reinterpret_cast<half2*>(pay16 + i) = __floats2half2_rn(a0, a1);
    }
  } else {
    for (int i = tid; i < a.n; i += T) {
      float acc = __half2float(quad_slot[0][i]);
      acc += __half2float(quad_slot[1][i]);
      acc += __half2float(quad_slot[2][i]);
      acc += __half2float(quad_slot[3][i]);
      staging[i] = acc;
    }
  }
  if (a.wire == 1) {
    // (5b) quantize to the wire encoding (per-128 group, rank-replicated).
    // Payload/scales live AFTER the staging region in the slot, so this
    // pass writes no byte still being read by any thread (no re-sync needed
    // for the data, only for the publish fence below).
    //
    // One WARP per 128-element group: 32 lanes x 4 elements, amax via 5
    // __shfl_xor_sync steps. The previous one-thread-per-group form did 256
    // serial element ops on a single lane (only n/128 of T threads active),
    // measured at ~104 ns/element vs the fp16 path's 0.3-1.2 -> +267 us per
    // call at the production 5,120 B size. Same scale and same payload bytes
    // as before: this is numerically identical.
    __syncthreads();
    {
      const int lane = (int)(tid & 31u);
      const int warp = (int)(tid >> 5u);
      const int nwarp = (int)(T >> 5u);
      const int ngroups = (a.n + 127) / 128;
      uint8_t* payload = own_region + payload_off;
      float* scales = (float*)(own_region + scales_off);
      for (int g = warp; g < ngroups; g += nwarp) {
        float v[4];
        float mx = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          v[j] = staging[g * 128 + lane * 4 + j];
          float av = v[j] < 0.f ? -v[j] : v[j];
          mx = av > mx ? av : mx;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        float sc = mx / 448.0f;
        if (sc == 0.0f) sc = 1.0f;
#pragma unroll
        for (int j = 0; j < 4; j++)
          payload[g * 128 + lane * 4 + j] = f32_to_e4m3(v[j] / sc);
        if (lane == 0) scales[g] = sc;
      }
    }
  } else if (a.wire == 2) {
    // (5b-nv4) NVFP4 quantize (per-16 block, rank-replicated):
    // s = e4m3(amax/6) (1 byte/block), payload = 2 e2m1 codes/byte
    // (even element low nibble). Wire = n/2 + n/16 bytes, sits in the
    // same 4n..5n payload zone (fits: 0.5625n < n).
    __syncthreads();
    uint8_t* payload = (uint8_t*)(own_region + payload_off);
    // n_max-anchored (NOT a.n): the reader computes the scale base as
    // pay + n_max/2 on the slot-stride n_max (same convention as the
    // e4m3 path's 5*n_max scales).
    uint8_t* scales4 = payload + (a.n_max >> 1);  // 1 byte per NV4_BLK elems
    for (int g = tid; g < a.n / NV4_BLK; g += T) {
      float x[16];
      float mx = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; j++) {
        x[j] = staging[g * 16 + j];
        float av = x[j] < 0.f ? -x[j] : x[j];
        if (av > mx) mx = av;
      }
      uint8_t sbyte = f32_to_e4m3(mx > 0.f ? __fdiv_rn(mx, 6.0f) : 1.0f);
      float s = e4m3_to_f32(sbyte);  // the scale as actually quantized
      float inv = (s > 0.f) ? __fdiv_rn(1.0f, s) : 1.0f;
      // 16 elements = 8 bytes; element 2j in the low nibble, 2j+1 in the
      // high nibble of byte j (matches the reader and the reference).
      uint8_t* pb = payload + g * 8;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        uint32_t c0 = nv4_e2m1_code(fabsf(x[2 * j]) * inv);
        uint32_t c1 = nv4_e2m1_code(fabsf(x[2 * j + 1]) * inv);
        if (x[2 * j] < 0.f && c0) c0 |= 0x8u;
        if (x[2 * j + 1] < 0.f && c1) c1 |= 0x8u;
        pb[j] = (uint8_t)(c0 | (c1 << 4));
      }
      scales4[g] = sbyte;
    }
  }
  // (5b-f16) wire=3 needs no post-pass: the half payload is already written by
  // the fused step-4 loop (each thread writes only its own elements, program
  // order), so the publish fence below covers it. Was: a second full pass
  // `pay16[i] = __float2half_rn(staging[i])` after writing staging in fp32.

  // (5) publish consumed[s] (clique ack) and partial_ready[s] (mirror
  // signal) CONCURRENTLY — the two release stores race each other; neither
  // waits on the other side's completion (1Cat: the clique completion is
  // published in parallel with the pair exchange).
  __syncthreads();
  // Device scope is sufficient here: every payload byte travels GPU-to-GPU
  // over NVLink into a peer's device memory (no host/pinned-memory path), and
  // the flag publish/consume pair is already system-scope
  // (st.release.sys / ld.acquire.sys). This is the NCCL convention for
  // intra-node device-to-device handshakes; __threadfence_system() ordered
  // host-visible memory we never touch and cost ~1.7 us per call (2 fences x
  // 98 calls/step = 0.34 ms, measured 78.0 -> 80.1 tok/s decode1).
  // Revert to __threadfence_system() if this kernel ever publishes to host
  // memory or crosses nodes.
  __threadfence();
  __syncthreads();
  if (tid == 0) st_flag_release(a.meta + 3 + s, E);  // consumed[s]
  if (tid == 1) st_flag_release(a.meta + 5 + s, E);
  unsigned long long t_quad = 0;
  if (ar_sample && threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_quad));
  }

  // (6) wait for the mirror's partial (acquire).
  if (tid == 2) {
    while (ld_flag_acquire(a.mirror_partial_ready + s) != E) {
    }
  }
  __syncthreads();

  // (7) total = partial(quad 0) + partial(quad 1) — the SAME expression (same
  // operands, same order) on all 8 ranks -> bitwise-equal outputs. In wire
  // mode both operands are dequantized from (identically encoded) payloads;
  // a quad-0 rank dequantizes its own payload for the quad-0 operand and the
  // mirror's payload for the quad-1 operand, and vice versa.
  const uint8_t* mr = a.mirror_region + s * region_stride;
  const float* val0 = (a.own_quad == 0) ? (const float*)own_region
                                        : (const float*)mr;
  const float* val1 = (a.own_quad == 0) ? (const float*)mr
                                        : (const float*)own_region;
  if (a.wire == 0) {
    for (int i = tid; i < a.n; i += T) {
      out[i] = __float2half_rn(val0[i] + val1[i]);
    }
  } else if (a.wire == 1) {
    // Wire layout per region slot: staging @0, payload @4*n_max (e4m3,
    // n_max bytes), scales @5*n_max (n_max/128 fp32) — the reader uses the
    // slot stride n_max (NOT a.n: the group scales were written by the
    // publisher at the n_max-based offsets).
    const uint8_t* pay0 = (a.own_quad == 0) ? own_region + 4 * a.n_max
                                           : mr + 4 * a.n_max;
    const uint8_t* pay1 = (a.own_quad == 0) ? mr + 4 * a.n_max
                                            : own_region + 4 * a.n_max;
    const float* sc0 = (const float*)(pay0 + a.n_max);
    const float* sc1 = (const float*)(pay1 + a.n_max);
    for (int i = tid; i < a.n; i += T) {
      int g = i >> 7;
      // Explicit __fmul_rn/__fadd_rn: under the production flags ptxas
      // contracts the second mul with the add into a single-rounding FMA,
      // deviating from the CPU model's separate-rounding convention on
      // ~0.1% of h2f-boundary elements (182/163840 at m=64). Full-TU
      // standalone (identical flags/data, no multi-rank): identical 182
      // mismatch set; -O2 reproduces, --fmad=false -> 0 (2026-10-15).
      float v0 = __fmul_rn(e4m3_to_f32(pay0[i]), sc0[g]);
      float v1 = __fmul_rn(e4m3_to_f32(pay1[i]), sc1[g]);
      out[i] = __float2half_rn(__fadd_rn(v0, v1));
    }
  } else if (a.wire == 2) {
    // NVFP4: payload @4*n_max (n/2 bytes, 2 codes/byte even-in-low),
    // scales @4*n_max + n/2 (1 e4m3 byte per 16 elems = per 8 payload
    // bytes). This tree's add convention: fp32 add, half cast (the old
    // tree's bf16-midstep is a host-staged convention, not ported).
    const uint8_t* pay0 = (a.own_quad == 0) ? own_region + 4 * a.n_max
                                           : mr + 4 * a.n_max;
    const uint8_t* pay1 = (a.own_quad == 0) ? mr + 4 * a.n_max
                                            : own_region + 4 * a.n_max;
    const uint8_t* sc0 = pay0 + (a.n_max >> 1);
    const uint8_t* sc1 = pay1 + (a.n_max >> 1);
    for (int i = tid; i < (a.n >> 1); i += T) {
      uint32_t c0l = pay0[i] & 0xFu, c0h = pay0[i] >> 4;
      uint32_t c1l = pay1[i] & 0xFu, c1h = pay1[i] >> 4;
      // 16 elements = 8 payload bytes per scale
      float s0 = e4m3_to_f32(sc0[i >> 3]);
      float s1 = e4m3_to_f32(sc1[i >> 3]);
      // Same FMA-contraction guard as the wire=1 reader (ptxas fuses a
      // trailing mul+add into one rounding, diverging from the CPU
      // model's separate-rounding convention on boundary elements).
      float v0l = __fmul_rn(nv4_e2m1_val(c0l), s0);
      float v0h = __fmul_rn(nv4_e2m1_val(c0h), s0);
      float v1l = __fmul_rn(nv4_e2m1_val(c1l), s1);
      float v1h = __fmul_rn(nv4_e2m1_val(c1h), s1);
      if (c0l & 8u) v0l = -v0l;
      if (c0h & 8u) v0h = -v0h;
      if (c1l & 8u) v1l = -v1l;
      if (c1h & 8u) v1h = -v1h;
      out[2 * i] = __float2half_rn(__fadd_rn(v0l, v1l));
      out[2 * i + 1] = __float2half_rn(__fadd_rn(v0h, v1h));
    }
  } else if (a.wire == 3) {
    // fp16 wire: [4n..6n) holds half (2*n_max B); no scales. Same
    // fp32-add + half-cast convention as every other wire mode; half2 loads
    // and one half2 store halve the instruction count of the merge pass.
    const half* pay0 = (a.own_quad == 0) ? (const half*)(own_region + payload_off)
                                         : (const half*)(mr + payload_off);
    const half* pay1 = (a.own_quad == 0) ? (const half*)(mr + payload_off)
                                         : (const half*)(own_region + payload_off);
    for (int i = 2 * tid; i < a.n; i += 2 * T) {
      const float2 f0 = __half22float2(*reinterpret_cast<const half2*>(pay0 + i));
      const float2 f1 = __half22float2(*reinterpret_cast<const half2*>(pay1 + i));
      *reinterpret_cast<half2*>(out + i) = __floats2half2_rn(f0.x + f1.x, f0.y + f1.y);
    }
  }

  // Sample 1 in 8 collectives: four L2 read-modify-writes on one cache line
  // cost ~1.8 us/call when done every time (measured -1.4% on decode), while
  // 12 samples/step still give an unbiased mean latency and byte rate.
  if (ar_sample && threadIdx.x == 0) {
    unsigned long long t_exit;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_exit));
    const unsigned long long dt = t_exit - t_entry;
    // Single writer per device (one CTA, one stream), so plain updates beat
    // atomics: the first version's four atomics cost 1% of a decode step.
    unsigned long long* st = &a.stats_buf[a.wire * 4];
    // One sample stands for eight collectives: weight every field, otherwise
    // mean latency and byte totals come out 8x low.
    st[0] += 8;
    st[1] += 8ull * dt;
    st[2] += 8ull * static_cast<unsigned long long>(a.n) * 2ull;
    if (dt > st[3]) {
      st[3] = dt;
    }
    if (t_quad != 0) {
      unsigned long long* hq = &a.stats_buf[16];
      const unsigned long long q = t_quad - t_entry;
      const unsigned long long mr = t_exit - t_quad;
      hq[0] += 8;
      hq[1] += q;
      if (q > hq[2]) hq[2] = q;
      unsigned long long* hm = &a.stats_buf[20];
      hm[0] += 8;
      hm[1] += mr;
      if (mr > hm[2]) hm[2] = mr;
    }
  }
}

// Boot probe kernel (1 thread): read my rank id (host-written into MY probe
// word) from the 4 P2P-verified legs — the 3 clique peers' probe words and
// the mirror's — and record ok=1 iff all match. This is the data-path IPC
// round-trip on the 4 cross-quad legs + 6 intra-quad legs (each leg probed
// from both ends across the 8 ranks).
//
// 8 SCALAR pointer parameters, NOT an array: at the <<<>>> launch site an
// array variable decays to a pointer, and the launch machinery marshals that
// pointer VALUE (a host stack address) into the parameter space — the first
// real GPU run measured exactly this: all 8 ranks' first global load faulted
// on 0x7ffc... host-stack addresses (compute-sanitizer, ar_smoke5), i.e. the
// kernel was dereferencing the ADDRESS OF the host rank_ids array, not its
// contents. Scalar parameters marshal by value — no decay path.
__global__ void sm70_hier_ar_probe_kernel(
    const int* __restrict__ p0, const int* __restrict__ p1,
    const int* __restrict__ p2, const int* __restrict__ p3,
    const int* __restrict__ p4, const int* __restrict__ p5,
    const int* __restrict__ p6, const int* __restrict__ p7,
    int rank, int c0, int c1, int c2, int mirror) {
  const int* rank_ids[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
  const int clique[3] = {c0, c1, c2};
  int ok = 1;
  if (rank_ids[rank][0] != rank) ok = 0;
  for (int k = 0; k < 3; k++) {
    int gr = clique[k];
    if (rank_ids[gr][0] != gr) ok = 0;
  }
  if (rank_ids[mirror][0] != mirror) ok = 0;
  int* words = (int*)rank_ids[rank];
  words[1] = ok;
}

// Device-side span timer (gate-1 G2/SKEW): 1-thread kernel storing the
// GPU-wide nanosecond timer (globaltimer(), SM35+) into a u64 slot.
// Why not CUDA events: events RECORDED DURING stream capture are
// unreadable from the host on this stack — cudaEventSynchronize/
// Query/ElapsedTime all return cudaErrorInvalidValue after replay
// (measured, ar_smoke6c). Plain kernel nodes have no such limit, so the
// in-graph AR sub-span is measured by bracketing the AR with two timer
// kernel nodes and reading the slots after replay.
__global__ void sm70_hier_ar_timer_kernel(uint64_t* slot) {
  uint64_t ns;
  // %globaltimer: GPU-wide nanosecond timer (sm_35+); not exposed as a
  // C++ intrinsic by the CUDA 12.8 headers — inline PTX.
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  *slot = ns;
}

class SM70HierAR {
 public:
  int rank_;
  int n_max_;
  void* base_;         // my IPC buffer base (allocated by Python via the
                       // house allocate_shared_buffer_and_handle op)
  uint32_t* meta_;
  half* in_buf_;
  uint8_t* region_;
  int* probe_words_;
  SM70HierARArgs args_;
  unsigned long long* stats_host_ = nullptr;
  unsigned long long* stats_dev_ = nullptr;
  // Bench-only sidecar override (VLLM_SM70_AR_LIB), idea-level from 1Cat's
  // CUSTOM_AR_LIBRARY env — NEVER a production path:
  //   extern "C" void sm70_hier_ar_reduce_override(
  //       void* fa, const half* in, half* out, int n, int wire,
  //       cudaStream_t stream);
  void* override_fn_ = nullptr;

  void allreduce(const half* in, half* out, int n, int wire,
                 cudaStream_t stream) {
    if (override_fn_) {
      typedef void (*OverrideFn)(void*, const half*, half*, int, int,
                                 cudaStream_t);
      auto fn = reinterpret_cast<OverrideFn>(override_fn_);
      fn(this, in, out, n, wire, stream);
      return;
    }
    SM70HierARArgs a = args_;
    a.n = n;
    a.wire = wire;
    static const bool stats_on = std::getenv("VLLM_SM70_AR_STATS") == nullptr ||
                                 std::getenv("VLLM_SM70_AR_STATS")[0] != '0';
    a.stats = stats_on;
    sm70_hier_ar_kernel<<<1, kThreads, 0, stream>>>(a, in, out);
    SM70_HIER_AR_CUDA_CHECK(cudaGetLastError());
  }
};

}  // namespace

// ----------------------------------------------------------------------------
// Host-side ops.
// ----------------------------------------------------------------------------
int64_t sm70_hier_ar_buf_size(int64_t max_bytes) {
  int64_t n = max_bytes / 2;  // fp16 elements
  n = (n + 127) & ~127LL;     // wire groups of 128
  if (n < 128) n = 128;
  return (int64_t)kMetaBytes + 4 * n + 12 * n + (int64_t)kProbeWords * 4;
}

int64_t sm70_hier_ar_n_max(int64_t max_bytes) {
  int64_t n = max_bytes / 2;
  n = (n + 127) & ~127LL;
  if (n < 128) n = 128;
  return n;
}

int64_t sm70_hier_ar_init(int64_t rank, int64_t max_bytes,
                          const std::vector<fptr_t>& peer_bases,
                          int64_t use_sidecar) {
  int world = (int)peer_bases.size();
  if (world != 8)
    throw std::invalid_argument("sm70_hier_ar: world size must be 8");
  if (rank < 0 || rank >= world)
    throw std::invalid_argument("sm70_hier_ar: invalid rank");
  int64_t n_max = sm70_hier_ar_n_max(max_bytes);
  auto fa = new (std::nothrow) SM70HierAR();
  if (!fa) throw std::bad_alloc();
  fa->rank_ = (int)rank;
  fa->n_max_ = (int)n_max;
  // Host-mapped telemetry counters: the kernel writes plain host memory, so any
  // process can poll them with no CUDA call and no stream, which is the only
  // scheme that survives CUDA graph capture (other-stream work is forbidden).
  {
    void* host_ptr = nullptr;
    if (cudaHostAlloc(&host_ptr, 24 * sizeof(unsigned long long),
                      cudaHostAllocMapped | cudaHostAllocPortable) ==
        cudaSuccess) {
      void* dev_ptr = nullptr;
      if (cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0) == cudaSuccess) {
        fa->stats_host_ = static_cast<unsigned long long*>(host_ptr);
        fa->stats_dev_ = static_cast<unsigned long long*>(dev_ptr);
        std::memset(fa->stats_host_, 0, 24 * sizeof(unsigned long long));
      }
    }
    fa->args_.stats_buf = fa->stats_dev_;
  }

  int quad0 = (rank < 4) ? 0 : 4;
  fa->args_.rank = (int)rank;
  fa->args_.own_quad = (quad0 >> 2);
  fa->args_.n_max = (int)n_max;
  fa->args_.n = 0;
  fa->args_.wire = 0;

  // Peer layout (boot-time: all IPC opens happened in Python, before any
  // graph capture).
  int peer_clique[3];
  int np = 0;
  for (int r = quad0; r < quad0 + 4; r++)
    if (r != rank) peer_clique[np++] = r;
  int mirror = 7 - rank;  // 0<->7, 1<->6, 2<->5, 3<->4
  for (int k = 0; k < 4; k++) {
    int gr = quad0 + k;
    char* b = reinterpret_cast<char*>(peer_bases[gr]);
    fa->args_.quad_in_buf[k] = reinterpret_cast<half*>(b + kMetaBytes);
  }
  for (int k = 0; k < 3; k++) {
    int gr = peer_clique[k];
    uint32_t* m = reinterpret_cast<uint32_t*>(peer_bases[gr]);
    fa->args_.peer_in_ready[k] = m + 1;
    fa->args_.peer_consumed[k] = m + 3;
  }
  uint32_t* mm = reinterpret_cast<uint32_t*>(peer_bases[mirror]);
  fa->args_.mirror_partial_ready = mm + 5;
  fa->args_.mirror_region =
      reinterpret_cast<uint8_t*>(peer_bases[mirror]) + kMetaBytes + 4 * n_max;

  // Own pointers.
  fa->base_ = reinterpret_cast<void*>(peer_bases[rank]);
  fa->meta_ = reinterpret_cast<uint32_t*>(fa->base_);
  fa->in_buf_ = reinterpret_cast<half*>(reinterpret_cast<char*>(fa->base_) +
                                        kMetaBytes);
  fa->region_ = reinterpret_cast<uint8_t*>(fa->in_buf_) + 4 * n_max;
  fa->probe_words_ = reinterpret_cast<int*>(fa->region_) + 3 * n_max;
  fa->args_.meta = fa->meta_;
  fa->args_.in_buf = fa->in_buf_;
  fa->args_.region = fa->region_;

  if (use_sidecar) {
    fa->override_fn_ = dlsym(RTLD_DEFAULT, "sm70_hier_ar_reduce_override");
  }
  return reinterpret_cast<fptr_t>(fa);
}

// Boot probe, phase 1 (host side): zero my meta words (cudaMalloc garbage
// would break the epoch-2 slot-reuse gate, which waits for the zeroed
// consumed flag; house pattern: custom_all_reduce.py L388-395 fills the
// buffer + syncs before peer-visible reads) and host-write my rank id into
// my probe word. The caller must dist.barrier() ALL ranks between this and
// sm70_hier_ar_probe_read — a rank's kernel reads its peers' probe words
// over IPC, which are only valid once every rank has run this phase.
void sm70_hier_ar_probe_write(int64_t _fa, int64_t rank) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  SM70_HIER_AR_CUDA_CHECK(cudaMemsetAsync(fa->meta_, 0, kMetaBytes, 0));
  // The protocol epoch must start at 1, not 0: every handshake flag is
  // zeroed above, so E=0 is indistinguishable from the initial state —
  // the step-3 wait (peers' in_ready[s] == E) would pass VACUOUSLY on the
  // first call (measured: smoke 6e G1 FAIL — arm B read stale in_buf slots
  // because the initial in_ready[0] == 0 == E already satisfied the wait),
  // and the E=2 slot-reuse gate (consumed[s] == prev = 0) would too.
  // Starting at 1 makes every wait real (flag 0 never equals E >= 1);
  // the slot-0 first use at E=2 keeps one harmless vacuous gate (no prior
  // occupant to overtake). Slot alternation (s = E & 1) is unchanged.
  uint32_t one = 1u;
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpyAsync(
      fa->meta_, &one, sizeof(uint32_t), cudaMemcpyHostToDevice, 0));
  int my_id = (int)rank;
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpyAsync(
      fa->probe_words_, &my_id, sizeof(int), cudaMemcpyHostToDevice, 0));
  SM70_HIER_AR_CUDA_CHECK(cudaStreamSynchronize(0));
}

// Boot probe, phase 2 (device side): 1-thread kernel reads every rank's probe
// word back through the 5 legs of my data path (local + 3 clique + mirror)
// and records ok = 1 iff all ids match. probe_bases = every rank's
// probe-word base (opened by Python).
void sm70_hier_ar_probe_read(int64_t _fa, int64_t rank,
                             const std::vector<fptr_t>& probe_bases) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  if (probe_bases.size() != 8)
    throw std::invalid_argument(
        "sm70_hier_ar_probe_read: 8 bases required");
  int clique[3];
  int quad0 = (rank < 4) ? 0 : 4;
  int np = 0;
  for (int r = quad0; r < quad0 + 4; r++)
    if (r != rank) clique[np++] = r;
  int mirror = 7 - rank;
  const int* rank_ids[8];
  for (int g = 0; g < 8; g++)
    rank_ids[g] = reinterpret_cast<const int*>(probe_bases[g]);
  // Scalar arguments only (no array parameters — see the kernel comment:
  // an array argument decays to its address at the launch site).
  sm70_hier_ar_probe_kernel<<<1, 1, 0, 0>>>(
      rank_ids[0], rank_ids[1], rank_ids[2], rank_ids[3], rank_ids[4],
      rank_ids[5], rank_ids[6], rank_ids[7], (int)rank, clique[0], clique[1],
      clique[2], mirror);
  SM70_HIER_AR_CUDA_CHECK(cudaDeviceSynchronize());
}

bool sm70_hier_ar_probe_ok(int64_t _fa) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  int ok = 0;
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(&ok, fa->probe_words_ + 1, sizeof(int),
                                     cudaMemcpyDeviceToHost));
  return ok == 1;
}

// Store globaltimer() (ns) into the 1-element slot (current stream —
// graph-capturable; used to bracket in-graph AR spans in the bench).
void sm70_hier_ar_timer(torch::stable::Tensor& slot) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      slot.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(slot.get_device_index());
  sm70_hier_ar_timer_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint64_t*>(slot.mutable_data_ptr()));
  SM70_HIER_AR_CUDA_CHECK(cudaGetLastError());
}

void sm70_hier_ar_reduce(int64_t _fa, torch::stable::Tensor& inp,
                         torch::stable::Tensor& out, int64_t wire) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());
  STD_TORCH_CHECK(inp.scalar_type() == torch::headeronly::ScalarType::Half,
                  "sm70_hier_ar: fp16 input required");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half,
                  "sm70_hier_ar: fp16 output required");
  STD_TORCH_CHECK(inp.numel() == out.numel(), "sm70_hier_ar: shape mismatch");
  STD_TORCH_CHECK(inp.is_contiguous(), "sm70_hier_ar: contiguous input required");
  STD_TORCH_CHECK(out.is_contiguous(), "sm70_hier_ar: contiguous output required");
  STD_TORCH_CHECK(inp.numel() % 128 == 0,
                  "sm70_hier_ar: numel must be a multiple of 128 (wire "
                  "groups)");
  STD_TORCH_CHECK(wire >= 0 && wire <= 3,
                  "sm70_hier_ar: wire must be 0 (fp32), 1 (e4m3), 2 (nvfp4) "
                  "or 3 (fp16)");
  STD_TORCH_CHECK(inp.numel() <= fa->n_max_,
                  "sm70_hier_ar: exceeds buffer cap");
  fa->allreduce(reinterpret_cast<const half*>(inp.const_data_ptr()),
                reinterpret_cast<half*>(out.mutable_data_ptr()),
                (int)inp.numel(), (int)wire, stream);
}

// Debug: copy the last-used slot's staging/payload/scales out of the own
// IPC region into caller-provided tensors (D2D; unit-only attribution).
// Layout per slot (stride 6*n_max bytes): staging @0 (n_max fp32),
// payload zone @4*n_max (e4m3: n_max bytes + fp32 scales @5*n_max;
// nvfp4: n_max/2 payload + scale bytes @4*n_max+n_max/2), fp16: 2*n_max.
int64_t sm70_hier_ar_stats_host_ptr(int64_t _fa) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  return reinterpret_cast<int64_t>(fa->stats_host_);
}

void sm70_hier_ar_stats_read(torch::stable::Tensor& out) {
  STD_TORCH_CHECK(out.is_cuda() && out.numel() >= 24,
                  "sm70_hier_ar_stats_read needs >= 24 CUDA elements");
  const torch::stable::accelerator::DeviceGuard device_guard(
      out.get_device_index());
  // The counters live in host-mapped memory now; this op is a convenience copy
  // for callers that want a CUDA tensor. Reading the host pointer directly
  // (sm70_hier_ar_stats_host_ptr) needs no CUDA call at all.
  STD_TORCH_CHECK(false, "use sm70_hier_ar_stats_host_ptr for host-mapped stats");
}

void sm70_hier_ar_dump(int64_t _fa, int64_t slot,
                       torch::stable::Tensor& stg, torch::stable::Tensor& pay,
                       torch::stable::Tensor& scf, torch::stable::Tensor& scb) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      stg.get_device_index());
  const int64_t n = fa->n_max_;
  const uint8_t* sl = fa->region_ + slot * 6 * n;
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(stg.mutable_data_ptr(), sl, 4 * n,
                                     cudaMemcpyDeviceToDevice));
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(pay.mutable_data_ptr(), sl + 4 * n, n,
                                     cudaMemcpyDeviceToDevice));
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(
      scf.mutable_data_ptr(), sl + 5 * n, 4 * ((n + 127) / 128),
      cudaMemcpyDeviceToDevice));
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(scb.mutable_data_ptr(),
                                     sl + 4 * n + (n >> 1), (n + 15) / 16,
                                     cudaMemcpyDeviceToDevice));
}

// Debug: same as sm70_hier_ar_dump but from the MIRROR's region (the
// reader's quad-1 operand source).
void sm70_hier_ar_dump_mr(int64_t _fa, int64_t slot,
                          torch::stable::Tensor& stg, torch::stable::Tensor& pay,
                          torch::stable::Tensor& scf, torch::stable::Tensor& scb) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      stg.get_device_index());
  const int64_t n = fa->n_max_;
  const uint8_t* sl = fa->args_.mirror_region + slot * 6 * n;
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(stg.mutable_data_ptr(), sl, 4 * n,
                                     cudaMemcpyDeviceToDevice));
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(pay.mutable_data_ptr(), sl + 4 * n, n,
                                     cudaMemcpyDeviceToDevice));
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(
      scf.mutable_data_ptr(), sl + 5 * n, 4 * ((n + 127) / 128),
      cudaMemcpyDeviceToDevice));
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(scb.mutable_data_ptr(),
                                     sl + 4 * n + (n >> 1), (n + 15) / 16,
                                     cudaMemcpyDeviceToDevice));
}

// Debug: copy quad member k's in_buf slot (the n_max halves the kernel's
// step-4 reads) into a caller-provided tensor.
void sm70_hier_ar_dump_quadin(int64_t _fa, int64_t k, int64_t slot,
                              torch::stable::Tensor& in_t) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      in_t.get_device_index());
  const int64_t n = fa->n_max_;
  const half* src = fa->args_.quad_in_buf[k] + slot * n;  // n_max per slot
  SM70_HIER_AR_CUDA_CHECK(cudaMemcpy(in_t.mutable_data_ptr(), src, 2 * n,
                                     cudaMemcpyDeviceToDevice));
}

void sm70_hier_ar_free(int64_t _fa) {
  auto fa = reinterpret_cast<SM70HierAR*>(_fa);
  if (fa) {
    delete fa;  // buffer ownership is Python's (house free_shared_buffer)
  }
}

// ----------------------------------------------------------------------------
// Stable torch library fragment (namespace _C_sm70_hier_ar; Python:
// torch.ops._C_sm70_hier_ar.<op>(...)).
// ----------------------------------------------------------------------------
STABLE_TORCH_LIBRARY_FRAGMENT(_C_sm70_hier_ar, sm70_hier_ar) {
  sm70_hier_ar.def("sm70_hier_ar_buf_size(int max_bytes) -> int");
  sm70_hier_ar.def("sm70_hier_ar_n_max(int max_bytes) -> int");
  sm70_hier_ar.def(
      "sm70_hier_ar_init(int rank, int max_bytes, int[] peer_bases, "
      "int use_sidecar) -> int");
  sm70_hier_ar.def("sm70_hier_ar_probe_write(int fa, int rank) -> ()");
  sm70_hier_ar.def(
      "sm70_hier_ar_probe_read(int fa, int rank, int[] probe_bases) -> ()");
  sm70_hier_ar.def("sm70_hier_ar_probe_ok(int fa) -> bool");
  sm70_hier_ar.def("sm70_hier_ar_timer(Tensor! slot) -> ()");
  sm70_hier_ar.def("sm70_hier_ar_stats_read(Tensor! out) -> ()");
  sm70_hier_ar.def("sm70_hier_ar_stats_host_ptr(int _fa) -> int");
  sm70_hier_ar.def(
      "sm70_hier_ar_reduce(int fa, Tensor inp, Tensor! out, int wire) -> ()");
  sm70_hier_ar.def(
      "sm70_hier_ar_dump(int fa, int slot, Tensor! stg, Tensor! pay, "
      "Tensor! scf, Tensor! scb) -> ()");
  sm70_hier_ar.def(
      "sm70_hier_ar_dump_mr(int fa, int slot, Tensor! stg, Tensor! pay, "
      "Tensor! scf, Tensor! scb) -> ()");
  sm70_hier_ar.def(
      "sm70_hier_ar_dump_quadin(int fa, int k, int slot, Tensor! in_t) -> ()");
  sm70_hier_ar.def("sm70_hier_ar_free(int fa) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_sm70_hier_ar, CUDA, sm70_hier_ar) {
  sm70_hier_ar.impl("sm70_hier_ar_reduce", TORCH_BOX(&sm70_hier_ar_reduce));
  sm70_hier_ar.impl("sm70_hier_ar_dump", TORCH_BOX(&sm70_hier_ar_dump));
  sm70_hier_ar.impl("sm70_hier_ar_dump_mr", TORCH_BOX(&sm70_hier_ar_dump_mr));
  sm70_hier_ar.impl(
      "sm70_hier_ar_dump_quadin", TORCH_BOX(&sm70_hier_ar_dump_quadin));
}

STABLE_TORCH_LIBRARY_IMPL(_C_sm70_hier_ar, CompositeExplicitAutograd,
                         sm70_hier_ar) {
  sm70_hier_ar.impl("sm70_hier_ar_buf_size", TORCH_BOX(&sm70_hier_ar_buf_size));
  sm70_hier_ar.impl("sm70_hier_ar_n_max", TORCH_BOX(&sm70_hier_ar_n_max));
  sm70_hier_ar.impl("sm70_hier_ar_init", TORCH_BOX(&sm70_hier_ar_init));
  sm70_hier_ar.impl("sm70_hier_ar_probe_write",
                    TORCH_BOX(&sm70_hier_ar_probe_write));
  sm70_hier_ar.impl("sm70_hier_ar_probe_read",
                    TORCH_BOX(&sm70_hier_ar_probe_read));
  sm70_hier_ar.impl("sm70_hier_ar_probe_ok", TORCH_BOX(&sm70_hier_ar_probe_ok));
  sm70_hier_ar.impl("sm70_hier_ar_timer", TORCH_BOX(&sm70_hier_ar_timer));
  sm70_hier_ar.impl("sm70_hier_ar_stats_read",
                   TORCH_BOX(&sm70_hier_ar_stats_read));
  sm70_hier_ar.impl("sm70_hier_ar_stats_host_ptr",
                   TORCH_BOX(&sm70_hier_ar_stats_host_ptr));
  sm70_hier_ar.impl("sm70_hier_ar_free", TORCH_BOX(&sm70_hier_ar_free));
}
