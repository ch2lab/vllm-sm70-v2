// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// KV dequant utilities for the in-tree QSA page4 kernels (grouped sparse
// page4 + XQA tc-256-wide), shared by qsa_page4_grouped.cu /
// qsa_page4_xqa.cu.
//
// Derived from the flashv100 lineage (pinned be3722bc, ownership decision
// 2026-10-16: in-tree local implementation) and extended with the NVFP4
// dtype (Track C in-service reader, production layout per
// nvfp4_kv_cache_kernels.cu: per-side page = [data H*B*128 | scale
// H*B*16], even element = low nibble).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace vllm {
namespace qsa_page4 {

constexpr int KV_CACHE_DTYPE_FP16 = 0;
constexpr int KV_CACHE_DTYPE_FP8_E4M3 = 1;
constexpr int KV_CACHE_DTYPE_FP8_E5M2 = 2;
constexpr int KV_CACHE_DTYPE_NVFP4 = 3;

__device__ __forceinline__ float quiet_nan_f() {
  return __int_as_float(0x7fffffff);
}

__device__ __forceinline__ float inf_f() { return __int_as_float(0x7f800000); }

__device__ __forceinline__ float exp2_int(const int exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
  const int sign = raw >> 7;
  const int exp = (raw >> 3) & 0x0f;
  const int mant = raw & 0x07;

  if ((raw & 0x7f) == 0) {
    return sign ? -0.0f : 0.0f;
  }

  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.001953125f;  // 2^-9
  } else {
    if (exp == 0x0f && mant == 0x07) {
      return quiet_nan_f();
    }
    value = (1.0f + static_cast<float>(mant) * 0.125f) * exp2_int(exp - 7);
  }
  return sign ? -value : value;
}

// NVFP4 (Track C): E2M1 4-bit code -> float. Codes 0..7 = {0, 0.5, 1, 1.5,
// 2, 3, 4, 6}; codes 8..15 are their negations. Bit-exact mirror of the
// in-tree nvfp4_ple.nvfp4_e2m1_lut (E2=(code>>1)&3, M1=code&1, sign=code>>3).
__device__ __forceinline__ float nvfp4_e2m1_to_float(int code) {
  const int sign = code >> 3;
  const int exp = (code >> 1) & 3;
  const int mant = code & 1;
  const float magnitude =
      (exp == 0) ? static_cast<float>(mant) * 0.5f
                 : (1.0f + static_cast<float>(mant) * 0.5f) * exp2_int(exp - 1);
  return sign ? -magnitude : magnitude;
}

// One packed data byte (even dim = low nibble, odd dim = high nibble, per
// the production store kernel) plus the e4m3 scale byte shared by that
// 16-element group -> {even, odd} halves.
__device__ __forceinline__ __half2 nvfp4_pair_to_half2(uint8_t data_byte,
                                                       uint8_t scale_byte) {
  const float scale = fp8_e4m3fn_to_float(scale_byte);
  const __half even =
      __float2half_rn(nvfp4_e2m1_to_float(data_byte & 0x0f) * scale);
  const __half odd =
      __float2half_rn(nvfp4_e2m1_to_float((data_byte >> 4) & 0x0f) * scale);
  return __half2(even, odd);
}

// ---- NVFP4 pool geometry (Track C reader; layout authority:
// nvfp4_kv_cache_kernels.cu store path, in-service pool from the allocator:
// [blocks, 2, N, 144] uint8, per-token cell [K data 128 | K scale 16 |
// V data 128 | V scale 16], K/V side bases 144 B apart, token stride
// 288 B, block stride = allocator page (may be padded past N*288)) ----
// Per (side, token) cell: 128 data B (D=256, 4-bit codes, even dim = low
// nibble) + 16 scale B (one e4m3 per 16 dims). A 4-token page at in-block
// page p: token p*4. The microblock table encodes pages_per_block per pool
// block (N/4, runtime), so a block's page table width is shape-driven, not
// stride-driven (allocator pages are not dense multiples of the cell).
constexpr int64_t kNvfp4TokenDataBytes = 128;
constexpr int64_t kNvfp4TokenScaleBytes = 16;
constexpr int64_t kNvfp4CellBytes =
    kNvfp4TokenDataBytes + kNvfp4TokenScaleBytes;  // 144 per side
constexpr int64_t kNvfp4TokenCellBytes =
    2 * kNvfp4CellBytes;  // 288: K+V cells per token

// 4 data bytes (8 dims, one panel uint4) + the e4m3 scale byte of their
// 16-dim group -> 8 halves packed as a uint4. The fp8 family converters
// emit half bit patterns; NVFP4 has no bit-cast trick, so build real
// __half2 values through the pair primitive.
__device__ __forceinline__ uint4 nvfp4_vector_to_half8(const uint32_t data_raw,
                                                       const uint8_t scale_byte) {
  const uint8_t* d = reinterpret_cast<const uint8_t*>(&data_raw);
  union {
    __half2 h[4];
    uint4 u;
  } r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r.h[i] = nvfp4_pair_to_half2(d[i], scale_byte);
  }
  return r.u;
}

// ---- NVFP4 shared-memory dequant LUTs (built once per CTA) ----
// Layout (kNvfp4LutBytes = 1536 B): first 256 x uint16_t is scale_lut,
// scale_lut[s] = __float2half_rn(fp8_e4m3fn_to_float(s)); next 256 x
// uint32_t is byte_lut, byte_lut[b] = half2(e2m1(b&15), e2m1(b>>4)) in
// fp16. The LUT reader below is bit-exact to the per-value path for all
// 16x256 combos: e4m3fn and e2m1 values map exactly into fp16 (the NaN
// scale byte is built through the same float path) and their product
// needs no fp16 rounding, so the RN half2 multiply equals
// __float2half_rn of the exact fp32 product.
constexpr int kNvfp4LutScaleEntries = 256;
constexpr int kNvfp4LutBytes =
    kNvfp4LutScaleEntries * 2 + kNvfp4LutScaleEntries * 4;

__device__ __forceinline__ void nvfp4_lut_build(uint16_t* __restrict__ lut,
                                               const int tid, const int nthr) {
  for (int s = tid; s < kNvfp4LutScaleEntries; s += nthr) {
    lut[s] = __half_as_ushort(__float2half_rn(fp8_e4m3fn_to_float(s)));
  }
  uint32_t* byte_lut =
      reinterpret_cast<uint32_t*>(lut + kNvfp4LutScaleEntries);
  for (int b = tid; b < kNvfp4LutScaleEntries; b += nthr) {
    union {
      __half2 h;
      uint32_t u;
    } e2;
    e2.h = __half2(__float2half_rn(nvfp4_e2m1_to_float(b & 0x0f)),
                   __float2half_rn(nvfp4_e2m1_to_float((b >> 4) & 0x0f)));
    byte_lut[b] = e2.u;
  }
}

// One data byte (2 nibbles) x one scale byte -> 2 halves. Two smem
// gathers + one RN half2 multiply; bit-exact to nvfp4_pair_to_half2.
__device__ __forceinline__ __half2 nvfp4_pair_from_lut(
    const uint8_t data_byte, const uint8_t scale_byte,
    const uint16_t* __restrict__ nvfp4_lut) {
  union {
    uint32_t u;
    __half2 h;
  } e2, s2, out;
  e2.u = reinterpret_cast<const uint32_t*>(
             nvfp4_lut + kNvfp4LutScaleEntries)[data_byte];
  const uint32_t s = nvfp4_lut[scale_byte];
  s2.u = s | (s << 16);
  out.h = __hmul2(e2.h, s2.h);
  return out.h;
}

__device__ __forceinline__ uint4 nvfp4_vector_to_half8_lut(
    const uint32_t data_raw, const uint8_t scale_byte,
    const uint16_t* __restrict__ nvfp4_lut) {
  const uint8_t* d = reinterpret_cast<const uint8_t*>(&data_raw);
  union {
    __half2 h[4];
    uint4 u;
  } r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r.h[i] = nvfp4_pair_from_lut(d[i], scale_byte, nvfp4_lut);
  }
  return r.u;
}

// Shared panel load for the grouped + XQA NVFP4 readers: panel column
// `vec_col` (absolute dims [panel_offset + 8*vec_col, + 7]) of the token
// at `token_offset` -> 8 halves. The page-table value encodes the pool
// block and the in-block 4-token page (microblock = pool_block *
// pages_per_block + page_in_block, pages_per_block = N/4 with N the pool
// block's token count); the harness hands the planner's p4 page table
// straight through, so the kernel derives both from the value. The pool
// page is a fixed 4-token unit, so the in-block position uses
// token_in_block = page_in_block * 4 + token_in_page and the template
// BLOCK_SIZE does not enter the addressing. The k/v base pointers already
// carry the per-side offset (K at the block start, V at +144).
template <int BLOCK_SIZE, bool NVFP4_LUT = false>
__device__ __forceinline__ uint4 load_nvfp4_pool_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int vec_col, const int token_offset, const int panel_offset,
    const int64_t block_stride, const int pages_per_block,
    const uint16_t* __restrict__ nvfp4_lut = nullptr) {
  const int logical_page = token_offset >> 2;
  const int token_in_page = token_offset & 3;
  const int physical_page = page_ids[logical_page];
  const int pool_block = physical_page / pages_per_block;
  const int page_in_block = physical_page - pool_block * pages_per_block;
  const int token_in_block = page_in_block * 4 + token_in_page;
  const int abs_dim = panel_offset + 8 * vec_col;
  const uint64_t base =
      static_cast<uint64_t>(pool_block) * static_cast<uint64_t>(block_stride) +
      static_cast<uint64_t>(token_in_block) * kNvfp4TokenCellBytes;
  const uint8_t* row_data =
      reinterpret_cast<const uint8_t*>(kv_cache) + base +
      static_cast<size_t>(abs_dim) / 2;
  const uint8_t* row_scale =
      reinterpret_cast<const uint8_t*>(kv_cache) + base +
      kNvfp4TokenDataBytes + static_cast<size_t>(abs_dim) / 16;
  if constexpr (NVFP4_LUT) {
    return nvfp4_vector_to_half8_lut(
        __ldg(reinterpret_cast<const uint32_t*>(row_data)), __ldg(row_scale),
        nvfp4_lut);
  }
  return nvfp4_vector_to_half8(
      __ldg(reinterpret_cast<const uint32_t*>(row_data)), __ldg(row_scale));
}

// 16-dim pair (two panel vectors sharing one scale byte): one 8B data
// load + one scale byte instead of two 4B + two 1B loads. lo = dims
// [abs_dim, +8), hi = dims [abs_dim+8, +16).
template <int BLOCK_SIZE, bool NVFP4_LUT = false>
__device__ __forceinline__ void load_nvfp4_pool_vector_pair(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int vec_pair, const int token_offset, const int panel_offset,
    const int64_t block_stride, const int pages_per_block, uint4& lo,
    uint4& hi, const uint16_t* __restrict__ nvfp4_lut = nullptr) {
  const int logical_page = token_offset >> 2;
  const int token_in_page = token_offset & 3;
  const int physical_page = page_ids[logical_page];
  const int pool_block = physical_page / pages_per_block;
  const int page_in_block = physical_page - pool_block * pages_per_block;
  const int token_in_block = page_in_block * 4 + token_in_page;
  const int abs_dim = panel_offset + 16 * vec_pair;
  const uint64_t base =
      static_cast<uint64_t>(pool_block) * static_cast<uint64_t>(block_stride) +
      static_cast<uint64_t>(token_in_block) * kNvfp4TokenCellBytes;
  const uint8_t* row_data =
      reinterpret_cast<const uint8_t*>(kv_cache) + base +
      static_cast<size_t>(abs_dim) / 2;
  const uint8_t* row_scale =
      reinterpret_cast<const uint8_t*>(kv_cache) + base +
      kNvfp4TokenDataBytes + static_cast<size_t>(abs_dim) / 16;
  const uint64_t data_raw = __ldg(reinterpret_cast<const uint64_t*>(row_data));
  const uint8_t scale_byte = __ldg(row_scale);
  if constexpr (NVFP4_LUT) {
    lo = nvfp4_vector_to_half8_lut(static_cast<uint32_t>(data_raw), scale_byte,
                                   nvfp4_lut);
    hi = nvfp4_vector_to_half8_lut(
        static_cast<uint32_t>(data_raw >> 32), scale_byte, nvfp4_lut);
  } else {
    lo = nvfp4_vector_to_half8(static_cast<uint32_t>(data_raw), scale_byte);
    hi = nvfp4_vector_to_half8(static_cast<uint32_t>(data_raw >> 32),
                               scale_byte);
  }
}

// One 8-byte data chunk (16 elements) + one scale byte -> 16 halves packed
// as two uint4: lo = elements 0..7, hi = elements 8..15.
__device__ __forceinline__ void nvfp4_decode_chunk(uint64_t data_raw,
                                                   uint8_t scale_byte,
                                                   uint4& lo, uint4& hi) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<__half2*>(&lo.x)[i] = nvfp4_pair_to_half2(
        static_cast<uint8_t>(data_raw >> (8 * (2 * i))), scale_byte);
    reinterpret_cast<__half2*>(&hi.x)[i] = nvfp4_pair_to_half2(
        static_cast<uint8_t>(data_raw >> (8 * (2 * i + 1))), scale_byte);
  }
}

__device__ __forceinline__ __half fp8_e5m2_to_half(uint8_t raw) {
  // E5M2 and IEEE fp16 use the same sign bit, exponent width, and exponent
  // bias. Expanding the two mantissa bits into the high fp16 mantissa bits is
  // exact for normals, subnormals, zero, inf, and NaN.
  return __ushort_as_half(static_cast<unsigned short>(raw) << 8);
}

// Finite E4M3 values map exactly to FP16 bits followed by FP32 scaling.
// Preserve the original NaN payload and signed zeros, including subnormals.
__device__ __forceinline__ float fp8_e4m3fn_to_float_bits(uint8_t raw) {
  const uint16_t half_bits = ((static_cast<uint16_t>(raw) << 7) & 0x3f80u) |
                             ((static_cast<uint16_t>(raw) << 8) & 0x8000u);
  const float value = __half2float(__ushort_as_half(half_bits)) * 256.0f;
  return (raw & 0x7fu) == 0x7fu ? quiet_nan_f() : value;
}

__device__ __forceinline__ __half2 fp8_e5m2_pair_to_half2(uint16_t raw_pair) {
  const uint32_t half2_bits = (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
                              (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = half2_bits;
  return converter.h2;
}

__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t raw) {
  return __half2float(fp8_e5m2_to_half(raw));
}

template <int KV_DTYPE, bool E4M3_BITS = false>
__device__ __forceinline__ float load_kv_cache_float_unscaled(
    const void* __restrict__ cache, const int64_t index) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP16) {
    const __half* cache_h = reinterpret_cast<const __half*>(cache);
    return __half2float(cache_h[index]);
  } else {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    const uint8_t raw = cache_u8[index];
    if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3 && E4M3_BITS) {
      return fp8_e4m3fn_to_float_bits(raw);
    }
    const float value = KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3
                            ? fp8_e4m3fn_to_float(raw)
                            : fp8_e5m2_to_float(raw);
    return value;
  }
}

template <int KV_DTYPE>
__device__ __forceinline__ float load_kv_cache_float(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  return load_kv_cache_float_unscaled<KV_DTYPE>(cache, index) * scale;
}

template <int KV_DTYPE>
__device__ __forceinline__ __half load_kv_cache_half(
    const void* __restrict__ cache, const int64_t index, const float scale) {
  if constexpr (KV_DTYPE == KV_CACHE_DTYPE_FP8_E5M2) {
    const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache);
    return __float2half_rn(__half2float(fp8_e5m2_to_half(cache_u8[index])) *
                           scale);
  }
  return __float2half_rn(load_kv_cache_float<KV_DTYPE>(cache, index, scale));
}

// ---- 8-byte panel vector conversions (uint4 in, 8 halves out) ----

__device__ __forceinline__ uint32_t
fp8_e5m2_pair_to_half2_bits(const uint16_t raw_pair) {
  return (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
         (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
}

__device__ __forceinline__ uint4 fp8_e5m2_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint16_t fp8_e4m3fn_to_half_bits(const uint8_t raw) {
  const uint16_t sign = static_cast<uint16_t>(raw & 0x80u) << 8;
  const uint8_t magnitude = raw & 0x7fu;
  const uint8_t exponent = magnitude >> 3;
  const uint8_t mantissa = magnitude & 0x07u;
  if (magnitude == 0) {
    return sign;
  }
  if (exponent == 0) {
    // E4M3 subnormals are exact fp16 normals: mantissa * 2^-9.
    const uint16_t magnitude_bits =
        mantissa < 2
            ? 0x1800u
            : (mantissa < 4
                   ? static_cast<uint16_t>(0x1c00u | ((mantissa - 2) << 9))
                   : static_cast<uint16_t>(0x2000u | ((mantissa - 4) << 8)));
    return sign | magnitude_bits;
  }
  if (magnitude == 0x7fu) {
    return sign | 0x7e00u;
  }
  return sign | static_cast<uint16_t>((exponent + 8) << 10) |
         static_cast<uint16_t>(mantissa << 7);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits(const uint16_t raw_pair) {
  return static_cast<uint32_t>(
             fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair))) |
         (static_cast<uint32_t>(
              fp8_e4m3fn_to_half_bits(static_cast<uint8_t>(raw_pair >> 8)))
          << 16);
}

__device__ __forceinline__ uint32_t
fp8_e4m3fn_pair_to_half2_bits_fast(const uint16_t raw_pair) {
  const uint8_t raw0 = static_cast<uint8_t>(raw_pair);
  const uint8_t raw1 = static_cast<uint8_t>(raw_pair >> 8);
  if ((raw0 & 0x7fu) == 0x7fu || (raw1 & 0x7fu) == 0x7fu) {
    return fp8_e4m3fn_pair_to_half2_bits(raw_pair);
  }

  // Moving a finite E4M3 encoding into the corresponding fp16 sign,
  // exponent, and mantissa fields represents exactly value / 256. A packed
  // half2 multiply restores both values without per-byte exponent branches.
  const uint32_t expanded = (static_cast<uint32_t>(raw_pair & 0x0080u) << 8) |
                            (static_cast<uint32_t>(raw_pair & 0x007fu) << 7) |
                            (static_cast<uint32_t>(raw_pair & 0x8000u) << 16) |
                            (static_cast<uint32_t>(raw_pair & 0x7f00u) << 15);
  union {
    uint32_t u;
    __half2 h2;
  } converter;
  converter.u = expanded;
  converter.h2 = __hmul2(converter.h2, __float2half2_rn(256.0f));
  return converter.u;
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4
fp8_e4m3fn_vector_to_half8_fast(const uint64_t raw) {
  return make_uint4(
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 16)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 32)),
      fp8_e4m3fn_pair_to_half2_bits_fast(static_cast<uint16_t>(raw >> 48)));
}

__device__ __forceinline__ uint4 fp8_e4m3fn_vector_to_half8_lut(
    const uint64_t raw, const uint16_t* __restrict__ lut) {
  return make_uint4(
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 8)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 16)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 24)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 32)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 40)]) << 16),
      static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 48)]) |
          (static_cast<uint32_t>(lut[static_cast<uint8_t>(raw >> 56)]) << 16));
}

}  // namespace qsa_page4
}  // namespace vllm
