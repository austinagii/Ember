#ifndef EMBER_AUTOGRAD_CONTEXT_H
#define EMBER_AUTOGRAD_CONTEXT_H

#include <ember/tensor_snapshot.h>

#include <vector>

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

  template <typename... Tensors>
  void save_for_backward(Tensors&... tensors) {
    auto _save_for_backward = [this](auto& tensor) {
      saved_tensors.emplace_back(tensor.save());
    };
    (_save_for_backward(tensors), ...);
  }
};

}  // namespace ember::autograd

#endif  // !EMBER_AUTOGRAD_CONTEXT_H
