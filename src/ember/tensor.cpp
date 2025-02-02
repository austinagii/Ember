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

Tensor::Tensor(double value) {
  data_ = xt::xarray<double>({value});
}

Tensor::Tensor(std::initializer_list<double> values) {
  data_ = xt::xarray<double>(values);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> values) {
  data_ = xt::xarray<double>(values);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<double>>> values) {
  data_ = xt::xarray<double>(values);
}

// TODO: Refactor this code so that performing a top sort of the graph happens within the engine.
void Tensor::backward() {
  // Create engine instance for this backward pass
  autograd::Engine engine;
  Tensor gradient;
  gradient.data_ = xt::ones_like(this->data_);
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

bool operator==(const Tensor& left, const Tensor& right) {
  return xt::all(xt::equal(left.data_, right.data_));
}

bool Tensor::equals_approx(const Tensor& other) {
  return xt::allclose(this->data_, other.data_);
}

Tensor Tensor::ones_like(const Tensor& other) {
  return Tensor::from_xarray_(xt::ones_like(other.data_));
}

Tensor Tensor::from_shape(std::initializer_list<size_t> shape) {
    return Tensor::from_xarray_(xt::xarray<double>::from_shape(shape));
}

} // namespace ember
