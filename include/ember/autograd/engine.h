#ifndef ENGINE_H
#define ENGINE_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>
#include <unordered_map>

namespace ember::autograd {
struct Engine {
  std::unordered_map<Node *, Tensor> grad_buffer;
  void evaluate_fn(Node *func);
};
}  // namespace ember::autograd

#endif  // !ENGINE_H
