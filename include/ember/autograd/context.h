#ifndef EMBER_AUTOGRAD_CONTEXT_H
#define EMBER_AUTOGRAD_CONTEXT_H

#include <ember/tensor_snapshot.h>

#include <vector>

namespace ember {
class Tensor;
}

namespace ember::autograd {

struct Context {
  /**
   * @brief Collection of tensor values captured during the forward pass.
   *
   * These snapshots can be used during the backward pass to compute accurate
   * gradients for operations that need access to the original input values
   * (e.g., division, power operations, etc.).
   */
  std::vector<ember::TensorSnapshot> saved_tensors;

  void save_for_backward(ember::Tensor& tensor);
};

}  // namespace ember::autograd

#endif  // !EMBER_AUTOGRAD_CONTEXT_H
