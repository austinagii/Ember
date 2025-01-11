#include <ember/tensor.h>
#include <ember/autograd/node.h> // Can this be removed?

#include <utility>

namespace ember {
  
Tensor::Tensor(): value(0.0f), gradient(nullptr), gradient_fn(nullptr), gradient_accumulator(nullptr) {}

Tensor::Tensor(float value): value(value), gradient(nullptr), gradient_fn(nullptr), gradient_accumulator(nullptr) {}

void Tensor::backward() {
  if (gradient == nullptr) {
    gradient = new Tensor(1.0f);
  }
  autograd::Engine::evaluate_fn(gradient_fn, *gradient);
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
  return std::move(TensorSnapshot(this));
}

TensorSnapshot::TensorSnapshot(const Tensor* tensor): value(tensor->value) {}

} // namespace ember
