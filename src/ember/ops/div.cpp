#include <ember/ops/div.h>
#include <ember/ops/utils.h>

#include <xtensor/xarray.hpp>

namespace ember {

std::size_t DIVIDEND_INDEX = 0;
std::size_t DIVISOR_INDEX = 1;

static Tensor div_forward(autograd::Context& context, Tensor& dividend,
                          Tensor& divisor) {
  if (xt::any(xt::equal(divisor.data_, 0.0))) {
    throw std::runtime_error("Division by zero is not allowed");
  }
  context.save_for_backward(dividend);
  context.save_for_backward(divisor);
  return Tensor::from_xarray_(xt::eval(dividend.data_ / divisor.data_));
}

/**
 * @brief Compute the gradients of both inputs of a division operation (a/b)
 *
 * Given that c = a / b and the partial derivate of c w.r.t some output o
 * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
 * follows:
 *
 * - ∂a/∂o = `output_grad` / b
 * - ∂b/∂o = -`output_grad` * a / (b * b)
 *
 * @param output_grad The gradient of the output of the division operation.
 * @return A vector containing the gradients of the inputs of the division
 * operation.
 */
std::vector<Tensor> div_backward(autograd::Context& context,
                                Tensor output_grad) {
  auto dividend = context.saved_tensors[DIVIDEND_INDEX];
  auto divisor = context.saved_tensors[DIVISOR_INDEX];

  auto dividend_grad_raw = xt::eval(output_grad.data_ / divisor.data_);
  auto divisor_grad_raw = xt::eval(
      output_grad.data_ * (-dividend.data_ / (divisor.data_ * divisor.data_)));

  auto dividend_grad =
      reduce_broadcast(dividend_grad_raw, dividend.data_.shape());
  auto divisor_grad = reduce_broadcast(divisor_grad_raw, divisor.data_.shape());

  return {Tensor::from_xarray_(dividend_grad),
          Tensor::from_xarray_(divisor_grad)};
}

REGISTER_BINARY_OP(div, div_forward, div_backward);

static Tensor divide_tensors(Tensor& dividend, Tensor& divisor) {
  return div(dividend, divisor);
}

Tensor operator/(Tensor& dividend, Tensor& divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor&& dividend, Tensor& divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor& dividend, Tensor&& divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor&& dividend, Tensor&& divisor) {
  return divide_tensors(dividend, divisor);
}

}  // namespace ember
