#include <ember/ops/matmul.h>
#include <ember/ops/utils.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>

namespace ember {

struct MatmulBackward;

Tensor matmul_forward(autograd::Context& ctx, const Tensor& a,
                      const Tensor& b) {
  ctx.save_for_backward(a);
  ctx.save_for_backward(b);
  return Tensor::from_xarray_(xt::linalg::dot(a.data_, b.data_));
}

std::vector<Tensor> matmul_backward(autograd::Context& ctx,
                                    const Tensor& output_grad) {
  return {Tensor::from_xarray_(xt::linalg::dot(
              output_grad.data_, xt::transpose(ctx.saved_tensors[1].data_))),
          Tensor::from_xarray_(xt::linalg::dot(
              xt::transpose(ctx.saved_tensors[0].data_), output_grad.data_))};
}

REGISTER_BINARY_OP(matmul, matmul_forward, matmul_backward)

}  // namespace ember
