#include <ember/ops/exp.h>
#include <ember/ops/utils.h>
#include <xtensor/xmath.hpp>

namespace ember {

Tensor exp_forward(autograd::Context& ctx, Tensor& exponent) {
  auto output = Tensor::from_xarray_(xt::exp(exponent.data_));
  ctx.save_for_backward(output);
  return output;
}

std::vector<Tensor> exp_backward(autograd::Context& ctx, Tensor output_grad) {
  return {Tensor::from_xarray_(ctx.saved_tensors[0].data_ * output_grad.data_)};
}

REGISTER_UNARY_OP(exp, exp_forward, exp_backward)

}  // namespace ember
