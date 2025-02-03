#ifndef EMBER_AUTOGRAD_ENGINE_H
#define EMBER_AUTOGRAD_ENGINE_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>
#include <unordered_map>

namespace ember::autograd {

/**
 * @brief The engine that performs backpropagation.
 *
 * This class is responsible for performing backpropagation on a computational
 * graph. It uses a topological sort to evaluate nodes in reverse topological
 * order.
 */
class Engine {
public:
  Engine() = default;
  ~Engine() = default;
  void backward(Node* root, Tensor gradient);

private:
  std::unordered_map<Node*, Tensor> grad_buffer;
  void evaluate_fn(Node* func, Tensor gradient);
};

}  // namespace ember::autograd

#endif  // !EMBER_AUTOGRAD_ENGINE_H
