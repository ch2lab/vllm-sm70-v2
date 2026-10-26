// SM70 (V100) native-FP8 GEMV/GEMM for per-tensor W8A8 linears.
//
// Why this exists: Volta has no FP8 hardware, so the only SM70 kernel that
// existed before this one dequantised the FP8 weight to FP16 once at load time
// and ran a plain 16-bit GEMM. That reads twice the bytes the checkpoint
// actually holds (Ornith's GDN + attention projections are ~0.64 GB of FP8 per
// rank that decode has to stream as 1.28 GB of FP16), and the dequantised copy
// used to sit next to the still-registered FP8 parameter.
//
// The GEMV keeps the weight in FP8 storage and converts in registers:
//   * the weight is [k, n] row-major, so a warp taking 128 consecutive columns
//     of one row is one fully coalesced 128-byte transaction, and each lane
//     owns 4 columns through a single uint32 load;
//   * E4M3 -> FP16 is pure bit movement (sign to bit 15, the 7 magnitude bits
//     up by 7 onto FP16's exponent and top mantissa bits) plus an exact 2^8
//     correction, so no lookup table and no shared memory;
//   * accumulation is FP32, and K is split across CTAs into FP32 partials that
//     a second kernel sums in a fixed order. Atomics are deliberately not used:
//     the SM70 Marlin split-K path reduces with atomicAdd into the FP16 output,
//     which is neither reproducible nor precise.
//
// Only m == 1 (single-token decode) uses the GEMV; larger m keeps the
// dequantise-to-FP16 + cuBLAS path, which is the right tool once a call has
// enough rows to amortise the weight read.
//
// STATUS (measured on V100-SXM2, /root/scratch/run/test_fp8_gemv_cuda.py):
// this kernel plateaus at ~260 GB/s (32.3 us for the 8.4 MB in_proj_qkv of
// one TP rank) and does NOT respond to wider per-lane loads (4 B -> 16 B) or
// to deeper load batching (2 -> 4 rows in flight) or to split-K beyond 32
// slices, so the ceiling is the per-byte E4M3 -> FP16 instruction stream, not
// memory parallelism. cuBLAS on the dequantised FP16 copy reads twice the
// bytes at ~646 GB/s (26 us), so this path is registered and tested but NOT
// wired into Sm70Fp8DequantLinearKernel. Beating FP16 cuBLAS needs the NVFP4
// QPN2 approach: an HMMA (mma.m8n8k4) operand feed with the fragments
// pre-packed, which converts each byte once into a register fragment instead
// of once per multiply.
//
// Registers op:
//   sm70_fp8_gemv(Tensor(a!) out, Tensor x, Tensor w_fp8, Tensor scale,
//                 int split_k) -> ()

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace {

constexpr int kColumnsPerWarp = 512;  // 16 bytes per lane * 32 lanes
constexpr int kBytesPerLane = 16;     // one uint4 per lane per K row
constexpr int kRowsPerStep = 4;       // independent uint4 loads in flight per lane

__device__ __forceinline__ uint32_t e4m3x2_to_half2(uint32_t b,
                                                     uint32_t shift) {
  // Two E4M3 bytes selected by `shift` (0 = low two bytes, 8 = high two)
  // expanded into one half2: the 7 magnitude bits move up by 7 (onto FP16's
  // exponent and top mantissa bits) and the sign bit moves up by 8. The
  // magnitude mask must exclude the sign bit, or the two terms overlap.
  const uint32_t sel = b >> shift;
  return ((sel & 0x007F007Fu) << 7) | ((sel & 0x00800080u) << 8);
}

__device__ __forceinline__ void unpack_e4m3x4(uint32_t b, half2 &lo,
                                              half2 &hi) {
  const half2 k256 = __float2half2_rn(256.0f);
  const uint32_t lo_pair = e4m3x2_to_half2(b, 0);
  const uint32_t hi_pair = e4m3x2_to_half2(b, 8);
  lo = __hmul2(*reinterpret_cast<const half2 *>(&lo_pair), k256);
  hi = __hmul2(*reinterpret_cast<const half2 *>(&hi_pair), k256);
}

// One E4M3 byte as the FP32 value it denotes. The bit move is exact (FP16
// holds every E4M3 value) and 2^8 is a power of two, so this is bit-for-bit the
// same number torch's FP8 -> FP32 cast produces.
__device__ __forceinline__ float e4m3_byte_to_float(uint32_t b) {
  const uint32_t bits = ((b & 0x7Fu) << 7) | ((b & 0x80u) << 8);
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits))) *
         256.0f;
}

template <int Warps>
__global__ void sm70_fp8_gemv_kernel(
    const uint8_t *__restrict__ w,  // [k, n] fp8 bytes, row-major in n
    const half *__restrict__ x,     // [k]
    half *__restrict__ out,         // [n] when split_k == 1
    float *__restrict__ partials,   // [split_k, n] otherwise
    const float *__restrict__ scale, int n, int k, int k_per_split) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // 16 bytes per lane: a warp then moves 512 B per K row, which is what it
  // takes to keep enough sectors in flight (4-byte lanes topped out at ~277
  // GB/s against cuBLAS' ~646 GB/s on the FP16 copy).
  const int col = (blockIdx.x * Warps + warp) * kColumnsPerWarp +
                  lane * kBytesPerLane;
  if (col >= n) {
    return;
  }
  const int k_begin = blockIdx.y * k_per_split;
  const int k_end = min(k_begin + k_per_split, k);
  const uint8_t *ptr = w + static_cast<size_t>(k_begin) * n + col;
  float acc[kBytesPerLane] = {};

  auto step = [&](const uint4 b, const half xv) {
    const half2 x2 = __half2half2(xv);
    const uint32_t *words = reinterpret_cast<const uint32_t *>(&b);
#pragma unroll
    for (int q = 0; q < kBytesPerLane / 4; ++q) {
      half2 lo, hi;
      unpack_e4m3x4(words[q], lo, hi);
      const float2 flo = __half22float2(__hmul2(lo, x2));
      const float2 fhi = __half22float2(__hmul2(hi, x2));
      // lo holds (byte 0, byte 2) and hi holds (byte 1, byte 3): the half2
      // lane order follows the mask pattern, not column order.
      acc[4 * q + 0] += flo.x;
      acc[4 * q + 2] += flo.y;
      acc[4 * q + 1] += fhi.x;
      acc[4 * q + 3] += fhi.y;
    }
  };

  int kk = k_begin;
  for (; kk + kRowsPerStep <= k_end; kk += kRowsPerStep) {
    uint4 b[kRowsPerStep];
    const uint8_t *q = ptr;
#pragma unroll
    for (int r = 0; r < kRowsPerStep; ++r) {
      b[r] = __ldcs(reinterpret_cast<const uint4 *>(q));
      q += n;
    }
#pragma unroll
    for (int r = 0; r < kRowsPerStep; ++r) {
      step(b[r], x[kk + r]);
    }
    ptr += static_cast<size_t>(kRowsPerStep) * n;
  }
  for (; kk < k_end; ++kk) {
    step(__ldcs(reinterpret_cast<const uint4 *>(ptr)), x[kk]);
    ptr += n;
  }

  const float s = __ldg(scale);
  if (partials != nullptr) {
    float *dst = partials + static_cast<size_t>(blockIdx.y) * n + col;
#pragma unroll
    for (int c = 0; c < kBytesPerLane; ++c) {
      dst[c] = acc[c] * s;
    }
  } else {
#pragma unroll
    for (int c = 0; c < kBytesPerLane; ++c) {
      out[col + c] = __float2half(acc[c] * s);
    }
  }
}

__global__ void sm70_fp8_gemv_reduce_kernel(const float *__restrict__ partials,
                                            half *__restrict__ out, int total,
                                            int splits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float4 *p4 = reinterpret_cast<const float4 *>(partials) + idx;
  float4 v = __ldg(p4);
  for (int s = 1; s < splits; ++s) {
    const float4 add =
        __ldg(reinterpret_cast<const float4 *>(partials +
                                             static_cast<size_t>(s) * total * 4) +
              idx);
    v.x += add.x;
    v.y += add.y;
    v.z += add.z;
    v.w += add.w;
  }
  reinterpret_cast<uint2 *>(out)[idx] =
      make_uint2(__half_as_ushort(__float2half_rn(v.x)) |
                     (__half_as_ushort(__float2half_rn(v.y)) << 16),
                 __half_as_ushort(__float2half_rn(v.z)) |
                     (__half_as_ushort(__float2half_rn(v.w)) << 16));
}

constexpr int kWarps = 1;

void sm70_fp8_gemv(torch::stable::Tensor &out, torch::stable::Tensor &x,
                   torch::stable::Tensor &w_fp8, torch::stable::Tensor &scale,
                   torch::stable::Tensor &partials, int64_t split_k) {
  const int n = static_cast<int>(w_fp8.size(1));
  const int k = static_cast<int>(w_fp8.size(0));
  STD_TORCH_CHECK(w_fp8.dim() == 2 && x.dim() == 1 && n % 16 == 0,
                  "SM70 FP8 GEMV needs a [k, n] weight with n % 16 == 0 (uint4 lane "
                  "loads) and a 1-D activation");
  STD_TORCH_CHECK(out.numel() == n, "SM70 FP8 GEMV output size mismatch");
  const int columns_per_cta = kWarps * kColumnsPerWarp;
  const int grid_x = (n + columns_per_cta - 1) / columns_per_cta;
  const int splits = static_cast<int>(split_k);
  STD_TORCH_CHECK(splits >= 1 && k % splits == 0,
                  "SM70 FP8 GEMV split_k must divide k");
  const int k_per_split = k / splits;
  cudaStream_t stream = get_current_cuda_stream(x.get_device_index());
  const uint8_t *w_ptr = reinterpret_cast<const uint8_t *>(w_fp8.data_ptr());
  const half *x_ptr = reinterpret_cast<const half *>(x.data_ptr());
  half *out_ptr = reinterpret_cast<half *>(out.mutable_data_ptr());
  const float *scale_ptr = scale.const_data_ptr<float>();

  if (splits == 1) {
    dim3 grid(grid_x, 1);
    sm70_fp8_gemv_kernel<kWarps><<<grid, 32 * kWarps, 0, stream>>>(
        w_ptr, x_ptr, out_ptr, nullptr, scale_ptr, n, k, k_per_split);
  } else {
    float *part_ptr =
        partials.defined() && partials.numel() >= static_cast<int64_t>(splits) * n
            ? reinterpret_cast<float *>(partials.mutable_data_ptr())
            : nullptr;
    STD_TORCH_CHECK(part_ptr != nullptr,
                    "SM70 FP8 GEMV needs a caller-provided fp32 partial buffer "
                    "of at least split_k * n elements when split_k > 1");
    dim3 grid(grid_x, splits);
    sm70_fp8_gemv_kernel<kWarps><<<grid, 32 * kWarps, 0, stream>>>(
        w_ptr, x_ptr, out_ptr, part_ptr, scale_ptr, n, k, k_per_split);
    const int total4 = n / 4;
    sm70_fp8_gemv_reduce_kernel<<<(total4 + 255) / 256, 256, 0, stream>>>(
        part_ptr, out_ptr, total4, splits);
  }
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 FP8 GEMV launch failed: ",
                  cudaGetErrorString(err));
}

}  // namespace


// One-pass FP8 -> FP16 activation dequantise: out[i] = round_fp16(fp32(x[i]) *
// scale). In Python this was `A.float() * As).to(fp16)`, which inductor splits
// into a cast kernel and a multiply-and-cast kernel per call (5.3 + 3.0 us for a
// single token), so the FP8 projections paid two launches and two passes over
// the activation per linear. Bit-identical to that expression: E4M3 -> FP16 is
// exact, the multiply is FP32 (__fmul_rn, no FMA contraction), and the round to
// FP16 happens once.
__global__ void sm70_fp8_dequant_act_kernel(const uint8_t* __restrict__ x,
                                            half* __restrict__ out,
                                            const float* __restrict__ scale,
                                            int total8) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total8) {
    return;
  }
  const float s = __ldg(scale);
  const uint2 word = reinterpret_cast<const uint2*>(x)[idx];
  const uint32_t* words = reinterpret_cast<const uint32_t*>(&word);
  uint4 result;
  half* halves = reinterpret_cast<half*>(&result);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    half2 lo, hi;
    unpack_e4m3x4(words[q], lo, hi);
    // unpack_e4m3x4 splits a word into the pairs (byte 0, byte 2) and
    // (byte 1, byte 3), so the four halves are written back in byte order.
    halves[4 * q + 0] = __float2half_rn(__fmul_rn(__low2float(lo), s));
    halves[4 * q + 1] = __float2half_rn(__fmul_rn(__low2float(hi), s));
    halves[4 * q + 2] = __float2half_rn(__fmul_rn(__high2float(lo), s));
    halves[4 * q + 3] = __float2half_rn(__fmul_rn(__high2float(hi), s));
  }
  reinterpret_cast<uint4*>(out)[idx] = result;
}

// Returns a fresh FP16 tensor rather than writing into a caller-provided one:
// a mutation on a buffer allocated inside vLLM's compiled region goes through
// torch._inductor's auto_functionalized wrapper, and with
// enable_auto_functionalized_v2=False the pass that is supposed to remove it
// aborts the whole compile ("AssertionError: auto_functionalized was not
// removed"), so the server fails to start.
torch::stable::Tensor sm70_fp8_dequant_act(torch::stable::Tensor& x_fp8,
                                           torch::stable::Tensor& scale) {
  const int64_t numel = x_fp8.numel();
  STD_TORCH_CHECK(numel % 8 == 0,
                  "SM70 FP8 activation dequantise needs a multiple of 8 elements");
  STD_TORCH_CHECK(x_fp8.is_contiguous(),
                  "SM70 FP8 activation dequantise needs a contiguous input");
  const auto in_sizes = x_fp8.sizes();
  std::vector<int64_t> shape(in_sizes.begin(), in_sizes.end());
  torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(), shape.size());
  auto out = torch::stable::new_empty(x_fp8, sizes,
                                      torch::headeronly::ScalarType::Half);
  const int total8 = static_cast<int>(numel / 8);
  cudaStream_t stream = get_current_cuda_stream(x_fp8.get_device_index());
  sm70_fp8_dequant_act_kernel<<<(total8 + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(x_fp8.data_ptr()),
      reinterpret_cast<half*>(out.mutable_data_ptr()),
      scale.const_data_ptr<float>(), total8);
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "SM70 FP8 activation dequantise launch failed: ",
                  cudaGetErrorString(err));
  return out;
}

// ---------------------------------------------------------------------------
// Fragment-fed FP8 GEMM: the same lane-major operand layout the NVFP4 QPN2
// kernel uses (one output column per lane, 16 consecutive K values per lane in
// `qpn2_logical_k` order, 32 lanes per 32-column tile, in-register expansion
// straight into the mma.m8n8k4 B fragment), but with an 8-bit payload instead
// of 4-bit codes.
//
// Why the scalar GEMV above cannot win and this can: the scalar form has to
// convert every weight element into a value it multiplies in FP32, so its cost
// per byte is dominated by the conversion instruction stream and it measured a
// ~260 GB/s ceiling regardless of lane width, prefetch depth or split-K. Here
// each byte is expanded once into the fragment register the tensor core
// consumes, exactly like the FP4 path that reaches ~690 GB/s on this box, and
// E4M3 -> FP16 is *exact* (all E4M3 values are representable in FP16), so this
// is numerically equivalent to the dequantise-to-FP16 + cuBLAS path it
// replaces while reading half the bytes and keeping no FP16 weight copy.
//
// The per-tensor weight scale is applied once to the accumulated FP32 result,
// which is the same arithmetic the FP16 path performs.

constexpr int kQpnRowsPerCta = 8;

__device__ __forceinline__ int qpn_col_from_lane(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

// Inverse of qpn_col_from_lane (a bijection on 0..31): the column's bits 3-1 go
// to the quad index, bit 0-1 to the in-quad row and bit 2 to lane bit 4.
__device__ __forceinline__ int qpn_lane_from_col(int col) {
  return (((col >> 3) & 3) << 2) | (col & 3) |
         (((col >> 2) & 1) << 4);
}

__device__ __forceinline__ int qpn_logical_k(int physical_k) {
  const int local = physical_k & 7;
  return (physical_k & 8) + ((local & 3) << 1) + (local >> 2);
}

// Byte order inside one 16-byte group. The B operand is consumed four MMAs at a
// time, each taking one uint32 = 4 weight bytes, which unpack_e4m3x4 splits into
// the half2 pair (byte 0, byte 2) and the pair (byte 1, byte 3). Those two
// half2s are the k-pairs {4q,4q+1} and {4q+2,4q+3} of that MMA, so slots 1 and 2
// must be swapped in the pack. (The FP4 path needs a different, nibble-level
// permutation for the same reason because a byte there holds two codes; see
// qpn_logical_k in sm70_nvfp4_qpn2.cu.)
__device__ __forceinline__ int qpn8_logical_k(int p) {
  return (p & ~3) | (((p & 1) << 1) | ((p >> 1) & 1));
}

__global__ void sm70_fp8_qpn_pack_kernel(const uint8_t *__restrict__ w,
                                         uint8_t *__restrict__ packed, int n,
                                         int k) {
  // Source is the FP8 linear's weight parameter: [k, n] row-major (the
  // orientation Fp8LinearMethod allocates, which is also what the FP16
  // dequantise path matmul'd against). Output is
  // [n/32 tile][k/16 group][lane][16 bytes], byte p holding the weight at
  // logical k = group*16 + qpn8_logical_k(p) of column
  // tile*32 + qpn_col_from_lane(lane). Element count is unchanged (n*k bytes),
  // writes are coalesced and the strided reads reuse L2 across neighbouring K
  // rows, so this is a load-time-cost-nothing transpose.
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * k;
  if (idx >= total) {
    return;
  }
  const int p = static_cast<int>(idx & 15);
  size_t outer = idx >> 4;
  const int lane = static_cast<int>(outer & 31);
  outer >>= 5;
  const int groups_k16 = k >> 4;
  const int group = static_cast<int>(outer % groups_k16);
  const int tile = static_cast<int>(outer / groups_k16);
  const int col = tile * 32 + qpn_col_from_lane(lane);
  const int k_logical = group * 16 + qpn8_logical_k(p);
  packed[idx] = w[static_cast<size_t>(k_logical) * n + col];
}

__global__ void sm70_fp8_qpn_dequant_kernel(const uint8_t *__restrict__ packed,
                                            half *__restrict__ out,
                                            const float *__restrict__ scale,
                                            int n, int k) {
  // Inverse of the pack, in the [k, n] orientation the FP16 cuBLAS path used
  // before this existed, so large-batch calls keep the arithmetic they had.
  // One thread per output element: coalesced FP16 stores, gathered byte loads
  // that hit L2 because consecutive K rows share 512-byte pack blocks.
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(n) * k;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % n);
  const int k_logical = static_cast<int>(idx / n);
  const int groups_k16 = k >> 4;
  const int tile = col >> 5;
  const int lane = qpn_lane_from_col(col & 31);
  const int group = k_logical >> 4;
  const int p = qpn8_logical_k(k_logical & 15);
  const uint8_t byte = packed[static_cast<size_t>(16) *
                                  (lane + 32 * (group + groups_k16 * tile)) +
                              p];
  out[idx] = __float2half(e4m3_byte_to_float(byte) * __ldg(scale));
}

#define SM70_FP8_QPN_MMA(C, A0, A1, B0, B1)                            \
  asm volatile(                                                        \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "               \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "               \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                   \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),    \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                             \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int SplitK, int NAcc, int RowTiles>
__global__ void sm70_fp8_qpn_kernel(const uint8_t *__restrict__ packed,
                                   const half *__restrict__ input,
                                   half *__restrict__ output,
                                   float *__restrict__ partials_out,
                                   const float *__restrict__ scale, int n, int k,
                                   int m, int ksplit) {
  constexpr int kBytesPerLaneGroup = 16;  // 16 K values, one byte each
  static_assert(kBytesPerLaneGroup == sizeof(uint4),
                "the pack hands each lane one uint4 per K group");
  __shared__ float partials[SplitK][RowTiles * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row_base = blockIdx.y * kQpnRowsPerCta * RowTiles;
  const int groups_k16 = k >> 4;
  const int groups_per_ksplit = groups_k16 / ksplit;
  const int groups_per_warp = groups_per_ksplit / SplitK;
  const int group_begin =
      blockIdx.z * groups_per_ksplit + warp * groups_per_warp;
  // One (tile, group) block is 32 lanes x 16 bytes = 512 bytes, and the pack
  // orders them [tile][group][lane].
  const uint4 *pack_ptr = reinterpret_cast<const uint4 *>(packed) +
                          (static_cast<size_t>(tile) * groups_k16 + 0) * 32 +
                          lane;
  const float s = __ldg(scale);

  float accum[RowTiles][NAcc][8];
#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][chain][i] = 0.0f;
      }
    }
  }

#pragma unroll 2
  for (int group = group_begin; group < group_begin + groups_per_warp;
       ++group) {
    // One uint4 per lane per group: 16 packed bytes = the group's 16 K values
    // for this lane's column, expanded into the 8 half2 B-fragment registers.
    const uint4 b0 = __ldcs(pack_ptr + static_cast<size_t>(group) * 32);
    half2 weights[8];
    const uint32_t *bytes = reinterpret_cast<const uint32_t *>(&b0);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      unpack_e4m3x4(bytes[q], weights[2 * q], weights[2 * q + 1]);
    }
    const unsigned *b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
      uint4 input01 = make_uint4(0, 0, 0, 0);
      uint4 input23 = make_uint4(0, 0, 0, 0);
      const int row = row_base + row_tile * kQpnRowsPerCta + local_row;
      if (row < m) {
        const half *input_row = input + static_cast<size_t>(row) * k;
        input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
      }
      const unsigned *a0 = reinterpret_cast<const unsigned *>(&input01);
      const unsigned *a1 = reinterpret_cast<const unsigned *>(&input23);
      SM70_FP8_QPN_MMA(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
      SM70_FP8_QPN_MMA(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
      SM70_FP8_QPN_MMA(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
      SM70_FP8_QPN_MMA(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        accum[row_tile][0][i] += accum[row_tile][chain][i];
      }
    }
  }

#pragma unroll
  for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int output_row = row_tile * kQpnRowsPerCta + (i & 2) +
                             ((lane & 16) ? 4 : 0) + (lane & 1);
      const int output_col =
          (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[warp][output_row * 32 + quadpair * 8 + output_col] =
          accum[row_tile][0][i];
    }
  }
  __syncthreads();

  for (int element = threadIdx.x; element < RowTiles * 256;
       element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
      value += partials[k_warp][element];
    }
    value *= s;
    const int output_row = row_base + (element >> 5);
    const int output_col = (tile << 5) + (element & 31);
    if (output_row < m) {
      const size_t out_idx = static_cast<size_t>(output_row) * n + output_col;
      if (ksplit == 1) {
        output[out_idx] = __float2half(value);
      } else {
        partials_out[static_cast<size_t>(blockIdx.z) * m * n + out_idx] = value;
      }
    }
  }
}

// Host side: pack + a GEMM launcher that mirrors the FP4 QPN2 dispatch, and a
// reduce over split-K partials laid out [split_k, m, n].
__global__ void sm70_fp8_qpn_reduce_kernel(const float *__restrict__ partials,
                                           half *__restrict__ out, int total,
                                           int ksplit) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  float value = 0.0f;
  for (int z = 0; z < ksplit; ++z) {
    value += partials[static_cast<size_t>(z) * total + idx];
  }
  out[idx] = __float2half(value);
}

// Persistent per-device fp32 scratch for the K-split partials ([ksplit, m, n]).
// It is reserved at pack time — weight load is always eager — because memory
// that is first allocated inside a CUDA-graph capture comes from that graph's
// private pool, and a pointer that escapes the pool cannot legally be shared
// with the other captured graphs. The GEMM keeps a guarded grow path for callers
// that pack somewhere else.
struct Fp8QpnState {
  torch::stable::Tensor buf;
};

Fp8QpnState& fp8_qpn_state(int device) {
  static std::vector<Fp8QpnState*> states;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (static_cast<int>(device) >= static_cast<int>(states.size())) {
    states.resize(device + 1, nullptr);
  }
  if (!states[device]) {
    states[device] = new Fp8QpnState();
  }
  return *states[device];
}

// Worst case this op allows: ksplit <= 8 rows of m <= 16 x n floats.
constexpr int64_t kFp8QpnMaxSplits = 8;
constexpr int64_t kFp8QpnMaxRows = 16;

void fp8_qpn_reserve(const torch::stable::Tensor& like, int64_t n) {
  auto& st = fp8_qpn_state(like.get_device_index());
  const int64_t need = kFp8QpnMaxSplits * kFp8QpnMaxRows * n;
  if (st.buf.defined() && st.buf.numel() >= need) {
    return;
  }
  std::vector<int64_t> shape = {need};
  torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(), shape.size());
  st.buf = torch::stable::new_empty(like, sizes,
                                    torch::headeronly::ScalarType::Float);
}

torch::stable::Tensor sm70_fp8_qpn_pack(torch::stable::Tensor& w_fp8) {
  torch::stable::accelerator::DeviceGuard device_guard(
      w_fp8.get_device_index());
  STD_TORCH_CHECK(w_fp8.dim() == 2,
                  "SM70 FP8 pack needs a [k, n] weight (FP8 linear layout)");
  const int64_t k = w_fp8.size(0);
  const int64_t n = w_fp8.size(1);
  STD_TORCH_CHECK(n % 32 == 0 && k % 16 == 0,
                  "SM70 FP8 pack needs n % 32 == 0 and k % 16 == 0, got n=", n,
                  " k=", k);
  std::vector<int64_t> shape = {n * k};
  torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(), shape.size());
  auto packed = torch::stable::new_empty(w_fp8, sizes,
                                         torch::headeronly::ScalarType::Byte);
  const size_t total = static_cast<size_t>(n) * k;
  cudaStream_t stream = get_current_cuda_stream(w_fp8.get_device_index());
  sm70_fp8_qpn_pack_kernel<<<static_cast<int>((total + 255) / 256), 256, 0,
                             stream>>>(
      reinterpret_cast<const uint8_t*>(w_fp8.data_ptr()),
      reinterpret_cast<uint8_t*>(packed.mutable_data_ptr()), static_cast<int>(n),
      static_cast<int>(k));
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 FP8 pack launch failed: ",
                  cudaGetErrorString(err));
  fp8_qpn_reserve(w_fp8, n);
  return packed;
}

torch::stable::Tensor sm70_fp8_qpn_dequant(torch::stable::Tensor& packed,
                                           torch::stable::Tensor& scale,
                                           int64_t k, int64_t n) {
  torch::stable::accelerator::DeviceGuard device_guard(
      packed.get_device_index());
  STD_TORCH_CHECK(packed.numel() == n * k,
                  "SM70 FP8 dequant: packed blob size mismatch");
  std::vector<int64_t> shape = {k, n};
  torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(), shape.size());
  auto out = torch::stable::new_empty(packed, sizes,
                                      torch::headeronly::ScalarType::Half);
  const size_t total = static_cast<size_t>(n) * k;
  cudaStream_t stream = get_current_cuda_stream(packed.get_device_index());
  sm70_fp8_qpn_dequant_kernel<<<static_cast<int>((total + 255) / 256), 256, 0,
                                stream>>>(
      reinterpret_cast<const uint8_t*>(packed.data_ptr()),
      reinterpret_cast<half*>(out.mutable_data_ptr()),
      scale.const_data_ptr<float>(), static_cast<int>(n), static_cast<int>(k));
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 FP8 dequant launch failed: ",
                  cudaGetErrorString(err));
  return out;
}

// Persistent per-device fp32 scratch for the K-split partials ([ksplit, m, n]).
// Grown lazily to the largest shape seen; warmup covers the max before
// CUDA-graph capture so the captured address is stable. A grow requested
// mid-capture is refused and the caller falls back to ksplit = 1.
// A/B knob: VLLM_SM70_FP8_QPN_KSPLIT caps the cross-CTA K-split.
// Unset = 1 (never split across CTAs). Measured on Ornith's per-rank shapes with
// 16 warps per CTA: a single fat CTA per 32-column tile beats splitting, on
// every shape - 757 GB/s on n=6144 and 385 on n=2048 against 625 / 315 once
// grid.z splits (which also adds a second reduce launch per call, worth
// 0.21 ms/step by itself). Splitting was the right answer for the 4-bit kernel,
// where each CTA holds a whole 16-value group in only 8 bytes per lane.
int fp8_qpn_ksplit_cap() {
  static int cap = -1;
  if (cap < 0) {
    const char* env = std::getenv("VLLM_SM70_FP8_QPN_KSPLIT");
    cap = (env != nullptr) ? std::atoi(env) : 1;
    if (cap < 1) cap = 1;
  }
  return cap;
}

void sm70_fp8_qpn_gemm(torch::stable::Tensor& out, torch::stable::Tensor& input,
                       torch::stable::Tensor& packed,
                       torch::stable::Tensor& scale, int64_t split_k) {
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const int n = static_cast<int>(out.size(1));
  const int k = static_cast<int>(input.size(1));
  const int m = static_cast<int>(input.size(0));
  STD_TORCH_CHECK(split_k == 8 || split_k == 16,
                  "SM70 FP8 QPN split_k must be 8 or 16");
  STD_TORCH_CHECK(n % 32 == 0 && k % 16 == 0,
                  "SM70 FP8 QPN needs n % 32 == 0 and k % 16 == 0");
  STD_TORCH_CHECK(input.stride(1) == 1 && out.stride(1) == 1,
                  "SM70 FP8 QPN needs row-contiguous input and output");
  const int gk = k / 16;
  STD_TORCH_CHECK(gk % static_cast<int>(split_k) == 0,
                  "SM70 FP8 QPN needs (k/16) % split_k == 0");
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  const uint8_t* packed_ptr =
      reinterpret_cast<const uint8_t*>(packed.data_ptr());
  const half* in_ptr = reinterpret_cast<const half*>(input.data_ptr());
  half* out_ptr = reinterpret_cast<half*>(out.mutable_data_ptr());
  const float* scale_ptr = scale.const_data_ptr<float>();

  int ksplit = 1;
  {
    const int nctas = n / 32;
    const int target = 5120 / static_cast<int>(split_k);
    const int cap = fp8_qpn_ksplit_cap();
    int want = (target + nctas - 1) / nctas;
    const int maxks = gk / static_cast<int>(split_k);
    if (want > cap) want = cap;
    if (want > 8) want = 8;
    if (want > maxks) want = maxks;
    for (int d = want; d > 1; --d) {
      if (gk % (static_cast<int>(split_k) * d) == 0) {
        ksplit = d;
        break;
      }
    }
  }
  float* part_ptr = nullptr;
  if (ksplit > 1) {
    // Normal case: sm70_fp8_qpn_pack already reserved the worst case this op
    // allows (ksplit <= 8, m <= 16), so nothing is allocated here.
    const int64_t need = kFp8QpnMaxSplits * kFp8QpnMaxRows * n;
    auto& st = fp8_qpn_state(input.get_device_index());
    if (!st.buf.defined() || st.buf.numel() < need) {
      cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &cap);
      if (cap == cudaStreamCaptureStatusNone) {
        fp8_qpn_reserve(input, n);
      } else {
        ksplit = 1;  // cannot grow the scratch inside a capture
      }
    }
    if (ksplit > 1) {
      part_ptr = reinterpret_cast<float*>(
          fp8_qpn_state(input.get_device_index()).buf.mutable_data_ptr());
    }
  }

#define LAUNCH_FP8_QPN(ROWS, SPLIT)                                          \
  do {                                                                       \
    const dim3 grid(n / 32, (m + kQpnRowsPerCta * ROWS - 1) /                \
                                 (kQpnRowsPerCta * ROWS),                    \
                    ksplit);                                                 \
    sm70_fp8_qpn_kernel<SPLIT, 1, ROWS><<<grid, 32 * SPLIT, 0, stream>>>(    \
        packed_ptr, in_ptr, out_ptr, part_ptr, scale_ptr, n, k, m, ksplit);  \
    if (ksplit > 1) {                                                        \
      const int total = m * n;                                               \
      sm70_fp8_qpn_reduce_kernel<<<(total + 255) / 256, 256, 0, stream>>>(   \
          part_ptr, out_ptr, total, ksplit);                                 \
    }                                                                        \
  } while (0)


  const bool two_tile = (m > kQpnRowsPerCta) && (m <= 16);
  STD_TORCH_CHECK(m <= 16,
                  "SM70 FP8 QPN covers at most 16 rows; larger batches use the "
                  "dequantise-to-FP16 path");
  if (two_tile && split_k == 8) {
    LAUNCH_FP8_QPN(2, 8);
  } else if (two_tile) {
    LAUNCH_FP8_QPN(2, 16);
  } else if (split_k == 8) {
    LAUNCH_FP8_QPN(1, 8);
  } else {
    LAUNCH_FP8_QPN(1, 16);
  }
#undef LAUNCH_FP8_QPN
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 FP8 QPN launch failed: ",
                  cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Single-op dispatch, mirroring sm70_nvfp4_gemm_dispatch: the batch-size
// decision lives HERE, not in Python, because the FP8 linears run inside
// vLLM's torch.compile region where `input.size(0)` is a symbolic integer - a
// Python `if m <= 16` there compiles into whichever branch the trace happened
// to take, and decode silently ends up on the prefill path.
struct Fp8PrefillState {
  torch::stable::Tensor w16;  // persistent FP16 weight scratch
  cublasHandle_t handle = nullptr;
};

Fp8PrefillState& fp8_prefill_state(int device) {
  static std::vector<Fp8PrefillState*> states;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (static_cast<int>(device) >= static_cast<int>(states.size())) {
    states.resize(device + 1, nullptr);
  }
  if (!states[device]) {
    auto* s = new Fp8PrefillState();
    cudaSetDevice(device);
    cublasCreate(&s->handle);
    states[device] = s;
  }
  return *states[device];
}

void sm70_fp8_qpn_dispatch(torch::stable::Tensor& out,
                           torch::stable::Tensor& input,
                           torch::stable::Tensor& packed,
                           torch::stable::Tensor& scale, int64_t split_k) {
  const int64_t m = input.size(0);
  const int64_t n = out.size(1);
  const int64_t k = input.size(1);
  if (m <= kFp8QpnMaxRows) {
    sm70_fp8_qpn_gemm(out, input, packed, scale, split_k);
    return;
  }
  // Large batches: expand the packed bytes into the persistent FP16 scratch
  // (bit-identical to the old load-time dequantise) and hand the GEMM to cuBLAS,
  // which is the right tool once a call has enough rows to amortise the weight
  // read. Reusing one scratch also keeps a per-layer FP16 copy out of the CUDA
  // graph pool.
  auto& st = fp8_prefill_state(input.get_device_index());
  if (!st.w16.defined() || st.w16.numel() < k * n) {
    std::vector<int64_t> shape = {k * n};
    torch::headeronly::IntHeaderOnlyArrayRef sizes(shape.data(), shape.size());
    st.w16 =
        torch::stable::new_empty(input, sizes, torch::headeronly::ScalarType::Half);
  }
  const int device = input.get_device_index();
  const cudaStream_t stream = get_current_cuda_stream(device);
  const size_t total = static_cast<size_t>(n) * k;
  sm70_fp8_qpn_dequant_kernel<<<static_cast<int>((total + 255) / 256), 256, 0,
                                stream>>>(
      reinterpret_cast<const uint8_t*>(packed.data_ptr()),
      reinterpret_cast<half*>(st.w16.mutable_data_ptr()),
      scale.const_data_ptr<float>(), static_cast<int>(n), static_cast<int>(k));
  // Row-major out[m,n] = input[m,k] @ w16[k,n] in cuBLAS column-major terms is
  // C'[n,m] = w16'[n,k] * input'[k,m] with both operands already transposed by
  // the layout, so no op flags and leading dimensions n / k / n.
  const float alpha = 1.0f, beta = 0.0f;
  cublasSetStream(st.handle, stream);
  cublasStatus_t gs = cublasGemmEx(
      st.handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha,
      st.w16.data_ptr(), CUDA_R_16F, static_cast<int>(n),
      input.data_ptr(), CUDA_R_16F, static_cast<int>(k), &beta,
      out.mutable_data_ptr(), CUDA_R_16F, static_cast<int>(n),
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  STD_TORCH_CHECK(gs == CUBLAS_STATUS_SUCCESS,
                  "SM70 FP8 QPN dispatch: cuBLAS gemm failed: ",
                  static_cast<int>(gs));
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "SM70 FP8 QPN dispatch launch failed: ",
                  cudaGetErrorString(err));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, sm70_fp8_gemv_ops) {
  sm70_fp8_gemv_ops.def(
      "sm70_fp8_gemv(Tensor(a!) out, Tensor x, Tensor w_fp8, Tensor scale, "
      "Tensor partials, int split_k) -> ()");
  sm70_fp8_gemv_ops.def("sm70_fp8_qpn_pack(Tensor w_fp8) -> Tensor");
  sm70_fp8_gemv_ops.def(
      "sm70_fp8_dequant_act(Tensor x_fp8, Tensor scale) -> Tensor");
  sm70_fp8_gemv_ops.def(
      "sm70_fp8_qpn_dequant(Tensor packed, Tensor scale, int k, int n) "
      "-> Tensor");
  sm70_fp8_gemv_ops.def(
      "sm70_fp8_qpn_gemm(Tensor(a!) out, Tensor input, Tensor packed, "
      "Tensor scale, int split_k) -> ()");
  sm70_fp8_gemv_ops.def(
      "sm70_fp8_qpn_dispatch(Tensor(a!) out, Tensor input, Tensor packed, "
      "Tensor scale, int split_k) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, sm70_fp8_gemv_ops) {
  sm70_fp8_gemv_ops.impl("sm70_fp8_gemv", TORCH_BOX(&sm70_fp8_gemv));
  sm70_fp8_gemv_ops.impl("sm70_fp8_qpn_pack", TORCH_BOX(&sm70_fp8_qpn_pack));
  sm70_fp8_gemv_ops.impl("sm70_fp8_dequant_act",
                         TORCH_BOX(&sm70_fp8_dequant_act));
  sm70_fp8_gemv_ops.impl("sm70_fp8_qpn_dequant",
                         TORCH_BOX(&sm70_fp8_qpn_dequant));
  sm70_fp8_gemv_ops.impl("sm70_fp8_qpn_gemm", TORCH_BOX(&sm70_fp8_qpn_gemm));
  sm70_fp8_gemv_ops.impl("sm70_fp8_qpn_dispatch",
                         TORCH_BOX(&sm70_fp8_qpn_dispatch));
}
