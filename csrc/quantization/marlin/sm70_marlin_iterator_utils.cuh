#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "quantization/marlin/sm70_marlin_common.cuh"

namespace marlin::sm70 {

template <int ContiguousWords>
struct QwordVector;

template <>
struct QwordVector<1> {
  using Type = uint32_t;
};

template <>
struct QwordVector<2> {
  using Type = uint2;
};

template <>
struct QwordVector<4> {
  using Type = uint4;
};

// Vector loads require 8/16-byte alignment, but the dense-tile layout's
// per-thread base word offset (intra-tile position) is not always aligned.
// Fall back to scalar 32-bit loads when the address is misaligned.
template <int ContiguousWords>
CUTLASS_DEVICE typename QwordVector<ContiguousWords>::Type load_qword_vector(
    uint32_t const* ptr) {
  static_assert(
      ContiguousWords == 1 || ContiguousWords == 2 || ContiguousWords == 4,
      "SM70 qword vector load supports 1, 2, or 4 words.");
  if constexpr (ContiguousWords == 1) {
    return *ptr;
  } else if constexpr (ContiguousWords == 2) {
    if ((reinterpret_cast<uintptr_t>(ptr) & 0x7) == 0) {
      return *reinterpret_cast<uint2 const*>(ptr);
    }
    uint2 out;
    out.x = ptr[0];
    out.y = ptr[1];
    return out;
  } else {
    if ((reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0) {
      return *reinterpret_cast<uint4 const*>(ptr);
    }
    uint4 out;
    out.x = ptr[0];
    out.y = ptr[1];
    out.z = ptr[2];
    out.w = ptr[3];
    return out;
  }
}

template <int ContiguousWords>
CUTLASS_DEVICE uint32_t qword_from_vector(
    typename QwordVector<ContiguousWords>::Type const& words, int c) {
  static_assert(
      ContiguousWords == 1 || ContiguousWords == 2 || ContiguousWords == 4,
      "SM70 qword vector access supports 1, 2, or 4 words.");
  if constexpr (ContiguousWords == 1) {
    return words;
  } else {
    uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
    return words_ptr[c];
  }
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint4 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint2 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

// 4-bit qword layout: within each K tile (kQuantTileK K values) the storage
// order is (local_k, local_n_vec, n_tile) with n_tile fastest (stride 1).
// Each word holds 8 consecutive N values (one local_n_vec) at one K. The MoE
// kernel's vectorized qword load reads the kContiguous c-words (one per
// 64-wide N tile) contiguously, so n_tile must be the fastest-varying
// dimension; this is unchanged for size_n == 64 (a single N tile).
template <int PackedMacroN>
CUTLASS_DEVICE int u4_packed_macro_n_qweight_offset_from_logical(
    int size_n, int logical_k, int logical_n) {
  static_assert(
      PackedMacroN == 64 || PackedMacroN == 128 || PackedMacroN == 256,
      "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const local_n_vec = (logical_n - n_tile * kQuantTileN) / 8;

  return k_tile * (size_n * 2) +
         local_k * (size_n / 8) +
         local_n_vec * (size_n / kQuantTileN) + n_tile;
}

template <int PackedMacroN>
CUTLASS_DEVICE int u8_packed_macro_n_qweight_offset_from_logical(
    int size_n, int logical_k, int logical_n) {
  static_assert(
      PackedMacroN == 64 || PackedMacroN == 128 || PackedMacroN == 256,
      "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const local_n_word = (logical_n - n_tile * kQuantTileN) / 4;

  return k_tile * (size_n * 4) +
         n_tile * (kQuantTileK * kQuantTileN / 4) +
         local_k * (kQuantTileN / 4) + local_n_word;
}

template <int PackedMacroN>
CUTLASS_DEVICE int u8_packed_macro_n_qweight_word_stride() {
  static_assert(
      PackedMacroN == 64 || PackedMacroN == 128 || PackedMacroN == 256,
      "SM70 Marlin packed macro-N must be 64, 128, or 256.");
  return PackedMacroN / kQuantTileN;
}

}  // namespace marlin::sm70
