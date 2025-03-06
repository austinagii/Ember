#include <ember/autograd/context.h>
#include <ember/tensor.h>

namespace ember::autograd {

void Context::save_for_backward(ember::Tensor& tensor) {
  saved_tensors.emplace_back(tensor.save());
}

}  // namespace ember