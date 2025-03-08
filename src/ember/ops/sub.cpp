#include <ember/ops/sub.h>
#include <ember/ops/utils.h>

#include <xtensor/xarray.hpp>

namespace ember {

std::size_t MINUEND_INDEX = 0;
std::size_t SUBTRAHEND_INDEX = 1;

static Tensor sub_forward(autograd::Context& context, Tensor& minuend,
                          Tensor& subtrahend) {
  context.save_for_backward(minuend);
  context.save_for_backward(subtrahend);
  return Tensor::from_xarray_(xt::eval(minuend.data_ - subtrahend.data_));
}

/**
 * @brief Compute the gradients of both inputs of a subtraction operation (a-b)
 *
 * Given that c = a - b and the partial derivate of c w.r.t some output o
 * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
 * follows:
 *
 * - ∂a/∂o = `output_grad` * 1
 * - ∂b/∂o = `output_grad` * -1
 *
 * @param output_grad The gradient of the output of the subtraction operation.
 * @return A vector containing the gradients of the inputs of the subtraction
 * operation.
 */
std::vector<Tensor> sub_backward(autograd::Context& context,
                                 Tensor output_grad) {
  auto minuend = context.saved_tensors[MINUEND_INDEX];
  auto subtrahend = context.saved_tensors[SUBTRAHEND_INDEX];

  xt::xarray<double> minuend_grad_broadcasted = output_grad.data_;
  xt::xarray<double> subtrahend_grad_broadcasted = -1 * output_grad.data_;

  xt::xarray<double> minuend_grad =
      reduce_broadcast(minuend_grad_broadcasted, minuend.data_.shape());
  xt::xarray<double> subtrahend_grad =
      reduce_broadcast(subtrahend_grad_broadcasted, subtrahend.data_.shape());

  return {Tensor::from_xarray_(minuend_grad),
          Tensor::from_xarray_(subtrahend_grad)};
}

REGISTER_BINARY_OP(sub, sub_forward, sub_backward);

}  // namespace ember
