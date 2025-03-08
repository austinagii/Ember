#include <ember/autograd/node.h>
#include <ember/ops/add.h>
#include <ember/ops/utils.h>
#include <ember/tensor.h>

#include <xtensor/xbroadcast.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>

namespace ember {

std::size_t AUGEND_INDEX = 0;
std::size_t ADDEND_INDEX = 1;

/**
 * @brief Add two tensors and return a new tensor representing the sum.
 */
static Tensor add_forward(autograd::Context& context, const Tensor& augend,
                          const Tensor& addend) {
  context.save_for_backward(augend, addend);
  return Tensor::from_xarray_(xt::eval(augend.data_ + addend.data_));
}

/**
 * @brief Compute the gradients of both inputs of an addition operation (a+b)
 *
 * Given that c = a + b and the partial derivate of c w.r.t some output o
 * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
 * follows:
 *
 * - ∂a/∂o = `output_grad` * 1
 * - ∂b/∂o = `output_grad` * 1
 *
 * @param output_grad The gradient of the output of the addition operation.
 * @return A vector containing the gradients of the inputs of the addition
 * operation.
 */
std::vector<Tensor> add_backward(autograd::Context& context,
                                 const Tensor& output_grad) {
  auto augend = context.saved_tensors[AUGEND_INDEX];
  auto addend = context.saved_tensors[ADDEND_INDEX];

  auto augend_gradient =
      reduce_broadcast(output_grad.data_, augend.data_.shape());
  auto addend_gradient =
      reduce_broadcast(output_grad.data_, addend.data_.shape());

  return {Tensor::from_xarray_(augend_gradient),
          Tensor::from_xarray_(addend_gradient)};
}

REGISTER_BINARY_OP(add, add_forward, add_backward);

}  // namespace ember
