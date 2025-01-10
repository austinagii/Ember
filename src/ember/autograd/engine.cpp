#include <ember/autograd/engine.h>
#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember::autograd {

void Engine::evaluate_fn(Node* func, Tensor output_grad) {
  if (func == nullptr) {
    return;
  }

  std::vector<Tensor> input_grads = (*func)(output_grad);
  if (input_grads.size() == 0) {
    return; // no gradients to propagate e.g. accumulator
  }

  if (input_grads.size() != func->next_fns.size()) {
    // TODO: Throw an appropriate error here.
  }
  for (std::size_t i = 0; i < func->next_fns.size(); i++) {
    evaluate_fn(func->next_fns[i], input_grads[i]);
  }
}

} // namespace ember::autograd
