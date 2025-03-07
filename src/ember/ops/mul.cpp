#include <ember/ops/mul.h>
#include <ember/ops/utils.h>

#include <xtensor/xarray.hpp>

namespace ember {

std::size_t MULTIPLICAND_INDEX = 0;
std::size_t MULTIPLIER_INDEX = 1;

static Tensor mul_forward(autograd::Context& context, Tensor& multiplicand,
                          Tensor& multiplier) {
  context.save_for_backward(multiplicand);
  context.save_for_backward(multiplier);
  return Tensor::from_xarray_(xt::eval(multiplicand.data_ * multiplier.data_));
}

/**
 * @brief Compute the gradients of both inputs of a multiplication operation (a*b)
 *
 * Given that c = a * b and the partial derivate of c w.r.t some output o
 * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
 * follows:
 *
 * - ∂a/∂o = `output_grad` * b
 * - ∂b/∂o = `output_grad` * a
 *
 * @param output_grad The gradient of the output of the multiplication operation.
 * @return A vector containing the gradients of the inputs of the multiplication
 * operation.
 */
std::vector<Tensor> mul_backward(autograd::Context& context,
                                Tensor output_grad) {
  auto multiplicand = context.saved_tensors[MULTIPLICAND_INDEX];
  auto multiplier = context.saved_tensors[MULTIPLIER_INDEX];

  xt::xarray<double> multiplicand_grad_raw =
      multiplier.data_ * output_grad.data_;
  xt::xarray<double> multiplier_grad_raw =
      multiplicand.data_ * output_grad.data_;

  auto multiplicand_grad =
      reduce_broadcast(multiplicand_grad_raw, multiplicand.data_.shape());
  auto multiplier_grad =
      reduce_broadcast(multiplier_grad_raw, multiplier.data_.shape());

  return {Tensor::from_xarray_(multiplicand_grad),
          Tensor::from_xarray_(multiplier_grad)};
}

REGISTER_BINARY_OP(mul, mul_forward, mul_backward);

static Tensor multiply_tensors(Tensor& multiplicand, Tensor& multiplier) {
  return mul(multiplicand, multiplier);
}

Tensor operator*(Tensor& multiplicand, Tensor& multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

Tensor operator*(Tensor&& multiplicand, Tensor& multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

Tensor operator*(Tensor& multiplicand, Tensor&& multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

Tensor operator*(Tensor&& multiplicand, Tensor&& multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

}  // namespace ember
