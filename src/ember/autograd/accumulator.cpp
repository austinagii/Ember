#include <ember/tensor.h>
#include <ember/autograd/accumulator.h>

#include <stdexcept>
#include <iostream>

namespace ember::autograd {

Accumulator::Accumulator(Tensor* target) : target(target) {
    if (!target) {
        throw std::invalid_argument("Accumulator target tensor cannot be nullptr");
    }
}

std::vector<Tensor> Accumulator::operator()(Tensor output_grad) {
  if (target->gradient == nullptr) {
    target->gradient = new Tensor(Tensor::zeros_like(*target));
  }
  target->gradient->data += output_grad.data;
  
  return {};
}

} // namespace ember::autograd
