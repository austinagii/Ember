#include <ember/ops/exp.h>

namespace ember {

Tensor exp(Tensor& exponent) {
  auto result = Tensor::from_xarray_(xt::exp(exponent.data_));

  result.requires_grad = exponent.requires_grad;
  if (result.requires_grad) {
    result.gradient_fn = new ExpBackward(exponent);
  }

  return result;
}

ExpBackward::ExpBackward(Tensor& exponent) : Node(exponent) {}

std::vector<Tensor> ExpBackward::operator()(Tensor output_grad) {
  return {Tensor::from_xarray_(xt::exp(saved_tensors[0].data_) * output_grad.data_)};
}

}  // namespace ember
