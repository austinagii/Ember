#include <ember/ops/div.h>

namespace ember {

std::size_t DIVIDEND_INDEX = 0;
std::size_t DIVISOR_INDEX = 1;

static Tensor divide_tensors(Tensor &dividend, Tensor &divisor) {
  if (xt::any(xt::equal(divisor.data_, 0.0))) {
    throw std::runtime_error("Division by zero is not allowed");
  }
  Tensor quotient =
      Tensor::from_xarray_(xt::eval(dividend.data_ / divisor.data_));

  if (dividend.requires_grad || divisor.requires_grad) {
    quotient.gradient_fn = new DivBackward(dividend, divisor);
    quotient.gradient_fn->saved_tensors.insert(
        quotient.gradient_fn->saved_tensors.begin(),
        {dividend.save(), divisor.save()});
    quotient.requires_grad = true;
  }
  return quotient;
}

Tensor operator/(Tensor &dividend, Tensor &divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor &&dividend, Tensor &divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor &dividend, Tensor &&divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor &&dividend, Tensor &&divisor) {
  return divide_tensors(dividend, divisor);
}

DivBackward::DivBackward(Tensor &dividend, Tensor &divisor) {
  if (dividend.requires_grad) {
    edges.push_back(autograd::Edge(0, dividend.get_gradient_edge()));
  }
  if (divisor.requires_grad) {
    edges.push_back(autograd::Edge(1, divisor.get_gradient_edge()));
  }
}

xt::xarray<float> reduce_broadcast(const xt::xarray<float> &grad,
                                   const xt::xarray<float> &original) {
  auto input_shape = original.shape();
  auto input_rank = input_shape.size();
  auto output_shape = grad.shape();
  auto output_rank = output_shape.size();

  // Align shapes on trailing dimensions and pad input dims if needed
  std::vector<std::size_t> padded_input_shape(output_rank, 1);
  for (std::size_t i = 0; i < input_rank; ++i) {
    padded_input_shape[output_rank - 1 - i] = input_shape[input_rank - 1 - i];
  }

  auto result = grad;
  // Summation from the highest dimension down to 0
  for (int dim = output_rank - 1; dim >= 0; dim--) {
    if (padded_input_shape[dim] != result.shape()[dim]) {
      result = xt::sum(result, dim);
    }
  }
  return result;
}

std::vector<Tensor> DivBackward::operator()(Tensor output_grad) {
  auto dividend = saved_tensors[DIVIDEND_INDEX];
  auto divisor = saved_tensors[DIVISOR_INDEX];

  // For division z = x/y:
  // ∂z/∂x = 1/y
  // ∂z/∂y = -x/y²

  // Calculate raw gradients with broadcasting
  auto dividend_grad_raw = xt::eval(output_grad.data_ / divisor.data_);
  auto divisor_grad_raw = xt::eval(
      output_grad.data_ * (-dividend.data_ / (divisor.data_ * divisor.data_)));

  // Reduce gradients along broadcast dimensions
  auto dividend_grad = reduce_broadcast(dividend_grad_raw, dividend.data_);
  auto divisor_grad = reduce_broadcast(divisor_grad_raw, divisor.data_);

  return {Tensor::from_xarray_(dividend_grad),
          Tensor::from_xarray_(divisor_grad)};
}

}  // namespace ember
