#include <ember/ops/matmul.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>

namespace ember {

struct MatmulBackward;

Tensor matmul(Tensor& a, Tensor& b) {
  auto dotprod = Tensor::from_xarray_(xt::linalg::dot(a.data_, b.data_));

  dotprod.requires_grad = (a.requires_grad || b.requires_grad);
  if (dotprod.requires_grad) {
    dotprod.gradient_fn = new MatmulBackward(a, b);
  }

  return dotprod;
}

MatmulBackward::MatmulBackward(Tensor& a, Tensor& b) : Node(a, b) {}

std::vector<Tensor> MatmulBackward::operator()(Tensor output_grad) {
  std::vector<Tensor> input_grads;

  input_grads.emplace_back(Tensor::from_xarray_(xt::linalg::dot(
      output_grad.data_, xt::transpose(saved_tensors[1].data_))));
  input_grads.emplace_back(Tensor::from_xarray_(xt::linalg::dot(
      xt::transpose(saved_tensors[0].data_), output_grad.data_)));

  for (auto& grad : input_grads) {
    std::cout << "Grad: " << grad.data_ << std::endl;
  }

  return input_grads;
}

}  // namespace ember
