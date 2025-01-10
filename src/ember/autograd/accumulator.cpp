#include <ember/tensor.h>
#include <ember/autograd/accumulator.h>

#include <stdexcept>

namespace ember::autograd {

Accumulator::Accumulator(Tensor* target) : target(target) {
    if (!target) {
        throw std::invalid_argument("Accumulator target tensor cannot be nullptr");
    }
}

std::vector<Tensor> Accumulator::operator()(Tensor output_grad) {
  if (target->gradient == nullptr) {
    target->gradient = new Tensor(0.0f);
  }
  target->gradient->value += output_grad.value;
  return {};
}

} // namespace ember::autograd
