#include <ember/tensor.h>
#include <ember/autograd/node.h> // Can this be removed?

#include <utility>
#include <vector>
#include <unordered_set>
#include <functional>
#include <iostream>

namespace ember {

Tensor::Tensor(): 
  gradient(nullptr), 
  gradient_fn(nullptr), 
  gradient_accumulator(nullptr) {}

Tensor::Tensor(xt::xarray<double> data): 
  data(data), 
  gradient(nullptr), 
  gradient_fn(nullptr), 
  gradient_accumulator(nullptr) {}

Tensor::Tensor(std::initializer_list<double> values) {
  data = xt::xarray<double>(values);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> values) {
  data = xt::xarray<double>(values);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<double>>> values) {
  data = xt::xarray<double>(values);
}

void Tensor::backward() {
  // Create engine instance for this backward pass
  autograd::Engine engine;
  Tensor gradient;
  gradient.data = xt::ones_like(this->data);
  engine.grad_buffer[gradient_fn] = gradient; 

  // Perform topological sort
  std::vector<autograd::Node*> topo_order;
  std::unordered_set<autograd::Node*> visited;
  
  // Helper function for DFS
  std::function<void(autograd::Node*)> topo_sort_dfs = [&](autograd::Node* node) {
    if (!node || visited.count(node)) return;
    
    visited.insert(node);
    for (const auto& edge : node->edges) {
      if (edge.fn) {
        topo_sort_dfs(edge.fn);
      }
    }
    topo_order.push_back(node);
  };
  
  // Start DFS from the gradient function
  topo_sort_dfs(gradient_fn);
  
  // Evaluate nodes in reverse topological order
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    engine.evaluate_fn(*it);
  }
}

autograd::Node* Tensor::get_gradient_edge() {
  if (gradient_fn != nullptr) {
    return gradient_fn;
  } 

  if (gradient_accumulator == nullptr) {
    gradient_accumulator = new autograd::Accumulator(this);
  }
  return gradient_accumulator;
}

TensorSnapshot Tensor::save() {
  return TensorSnapshot(this);
}

} // namespace ember
