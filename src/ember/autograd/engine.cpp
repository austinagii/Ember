#include <ember/autograd/engine.h>
#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember::autograd {

void Engine::evaluate_fn(Node* func, Tensor output_grad) {
  std::vector<Tensor> input_grads = (*func)(output_grad);

  if (input_grads.size() != func->get_num_inputs()) {
    throw std::runtime_error("Number of input gradients does not match number of inputs");
  }

  for (std::size_t i = 0; i < func->edges.size(); i++) {
    Edge edge = func->edges[i];
    evaluate_fn(edge.fn, input_grads[edge.input_nr]);
  }
}

} // namespace ember::autograd
