// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Geometry constants shared by the in-tree QSA page4 kernels (planner,
// grouped sparse page4, XQA tc-256-wide). Values are the TP8 production
// geometry (kGroupedVerifyHeads=4 / Rows=32, C-2 M1a build), pinned from
// the flashv100 lineage (be3722bc; ownership decision 2026-10-16: in-tree
// local implementation).

#pragma once

#include <cstdint>

namespace vllm {
namespace qsa_page4 {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

// ---- XQA tc-256-wide ----
constexpr int kXQATCBlockN = 128;
constexpr int kXQATCStride = 128;
constexpr int kXQATCPageIdsCapacity = kXQATCBlockN / 4;
constexpr int kXQATC256WideWarpCount = 8;
constexpr int kXQATC256WideThreads = kXQATC256WideWarpCount * kWarpSize;
constexpr int kXQATC256WideBlockM = 8;
constexpr int kXQATC256WideThreadsPerRow = kWarpSize;
constexpr int kXQATCG6DualCtaThreads = 6 * kWarpSize;
constexpr int kXQATCG6Pipeline8WarpThreads = 8 * kWarpSize;
constexpr int kXQARouteAllSeqLens = 0;
constexpr int kXQARouteShortSeqLens = -1;
constexpr int kXQARouteLongSeqLens = 1;
constexpr int kXQARouteP1024Sawtooth = 4;
constexpr int kXQARouteP256Sawtooth = 5;
constexpr int kXQARouteP1024SawtoothMid = 6;
constexpr int kXQARouteP1024SawtoothFinal = 7;
constexpr int kXQARouteRangeSeqLens = 8;
constexpr int kXQARouteWaveLongSeqLens = 9;
constexpr int kXQATC256WidePaddedQStride = 264;
constexpr int kXQATC256WidePaddedKVStride = 136;
constexpr int kXQATC256WideAlignedPaddedQStride = 272;
constexpr int kXQATC256WideAlignedPaddedKVStride = 144;
constexpr int kXQATCQKPipelinePanelDim = 64;
constexpr int kXQATCQKPipelineKVStride = 72;
constexpr int kXQATCStagedPVTileRows = 64;

// ---- Grouped verify (TP8 production geometry) ----
constexpr int kGroupedVerifyQ8MaxQ = 8;
constexpr int kGroupedVerifyQ16MaxQ = 16;
constexpr int kGroupedVerifyMaxSupportedQ = kGroupedVerifyQ16MaxQ;
constexpr int kGroupedVerifyHeads = 4;
constexpr int kGroupedVerifyHeadDim = 256;
constexpr int kGroupedVerifyRows = 32;
constexpr int kGroupedVerifyBlockN = 32;
constexpr int kGroupedVerifyQStride = 264;
constexpr int kGroupedVerifyKVStride = 264;
constexpr int kGroupedVerifyScoreStride = 32;
constexpr int kGroupedVerifyProbStride = 40;
constexpr int kGroupedVerifyPageIdsCapacity = 16;
// One packed head group times 80 context splits maps one 512-thread CTA to
// each of the V100's 80 SMs.
constexpr int kGroupedVerifyQ8Splits = 80;
constexpr int kGroupedVerifyWorkspaceRows =
    kGroupedVerifyQ8MaxQ * kGroupedVerifyQ8Splits;
constexpr int kGroupedVerifyMinTokensPerSplit = 64;
constexpr int kGroupedVerifySingleQueryMinTokensPerSplit = 128;
constexpr int kGroupedVerifyShortContextMaxTokens = 128;
constexpr int kGroupedVerifyThreads = 512;
constexpr int kGroupedVerifyWarps = kGroupedVerifyThreads / kWarpSize;
constexpr int kGroupedVerifyQKWarps =
    (kGroupedVerifyRows / 16) * (kGroupedVerifyBlockN / 16);

}  // namespace qsa_page4
}  // namespace vllm
