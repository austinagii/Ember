#include <ember/autograd/engine.h>
#include <ember/autograd/node.h>
#include <ember/tensor.h>
#include <unordered_map>

#include <iostream>

namespace ember::autograd {

void Engine::evaluate_fn(Node *func) {
  std::vector<Tensor> input_grads = (*func)(grad_buffer[func]);

  if (input_grads.size() < func->get_num_inputs()) {
    throw std::runtime_error(
        "Not enough gradients computed given the number of inputs");
  }

  // Store or accumulate gradients in buffer for next nodes
  for (std::size_t i = 0; i < func->edges.size(); i++) {
    Edge &edge = func->edges[i];

    auto it = grad_buffer.find(edge.fn);
    if (it == grad_buffer.end()) {
      grad_buffer[edge.fn] = input_grads[edge.input_nr];
    } else {
      grad_buffer[edge.fn] = grad_buffer[edge.fn] + input_grads[edge.input_nr];
    }
  }
}

}  // namespace ember::autograd
