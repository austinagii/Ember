#include <ember/autograd/engine.h>
#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace ember::autograd {

/**
 * Performs a topological sort of the computational graph from the given root.
 */
std::vector<Node*> topsort(Node* root) {
  std::vector<Node*> sorted;
  std::unordered_set<Node*> visited;

  std::function<void(Node*)> _topsort = [&](Node* node) {
    if (node == nullptr || visited.find(node) != visited.end()) {
      return;
    }

    visited.insert(node);
    for (const auto& edge : node->edges) {
      if (edge.fn) {
        _topsort(edge.fn);
      }
    }
    sorted.push_back(node);
  };
  _topsort(root);

  return sorted;
}

void Engine::backward(Node* root, Tensor gradient) {
  grad_buffer[root] = gradient;

  std::vector<Node*> nodes = topsort(root);
  // Evaluate nodes in reverse topological order
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    evaluate_fn(*it, grad_buffer[*it]);
  }
}

/**
 * Evaluate the backward function represented by the given node and store it's
 * input gradients in the gradient buffer.
 */
void Engine::evaluate_fn(Node* func, Tensor gradient) {
  std::vector<Tensor> input_grads = (*func)(gradient);

  if (input_grads.size() < func->get_num_inputs()) {
    throw std::runtime_error(
        "Not enough gradients computed given the number of inputs");
  }

  // Store or accumulate gradients in buffer for next nodes
  for (auto i = 0; i < func->edges.size(); i++) {
    Edge& edge = func->edges[i];

    // Register a new gradient for the edge node if none exists or add the
    // calculated edge to the previous gradient.
    auto it = grad_buffer.find(edge.fn);
    if (it == grad_buffer.end()) {
      grad_buffer[edge.fn] = input_grads[edge.input_nr];
    } else {
      grad_buffer[edge.fn] = grad_buffer[edge.fn] + input_grads[edge.input_nr];
    }
  }
}

}  // namespace ember::autograd
