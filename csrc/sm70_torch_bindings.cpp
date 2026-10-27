// SM70 additions to the stable-ABI op bindings.
//
// qsa_lexicographic_topk was registered in the deployment tree's central
// libtorch_stable/torch_bindings.cpp, which upstream deleted when it moved
// csrc/ to its current layout; the synced tree therefore had the kernels but no
// registration, and the engine died with
//   AttributeError: '_OpNamespace' '_C' object has no attribute
//   'qsa_lexicographic_topk'
// at kernel warmup. Keeping them in a file of our own keeps the next upstream
// sync from having to re-resolve the central bindings.

#include "core/registration.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

void qsa_lexicographic_topk(const torch::stable::Tensor& logits,
                            const torch::stable::Tensor& lengths,
                            torch::stable::Tensor& output, int64_t k);

void qsa_prefilter_topk(const torch::stable::Tensor& logits,
                        const torch::stable::Tensor& tile_max,
                        const torch::stable::Tensor& lengths,
                        torch::stable::Tensor& tiles,
                        torch::stable::Tensor& candidates,
                        torch::stable::Tensor& positions,
                        torch::stable::Tensor& tile_lengths,
                        torch::stable::Tensor& span_lengths,
                        torch::stable::Tensor& output, int64_t tile, int64_t k);

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "qsa_lexicographic_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "int k) -> ()");
  ops.def(
      "qsa_prefilter_topk(Tensor logits, Tensor tile_max, Tensor lengths, "
      "Tensor! tiles, Tensor! candidates, Tensor! positions, "
      "Tensor! tile_lengths, Tensor! span_lengths, Tensor! output, "
      "int tile, int k) -> ()");

  ops.impl("qsa_lexicographic_topk", TORCH_BOX(&qsa_lexicographic_topk));
  ops.impl("qsa_prefilter_topk", TORCH_BOX(&qsa_prefilter_topk));
}
