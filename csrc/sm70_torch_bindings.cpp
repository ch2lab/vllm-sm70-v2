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

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "qsa_lexicographic_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "int k) -> ()");

  ops.impl("qsa_lexicographic_topk", TORCH_BOX(&qsa_lexicographic_topk));
}
