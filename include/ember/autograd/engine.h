#ifndef ENGINE_H
#define ENGINE_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>
#include <unordered_map>

namespace ember::autograd {
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

#endif  // !ENGINE_H
