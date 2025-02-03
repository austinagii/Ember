#include <ember/ops/mul.h>
#include <ember/ops/utils.h>

namespace ember {

std::size_t MULTIPLICAND_INDEX = 0;
std::size_t MULTIPLIER_INDEX = 1;

static Tensor multiply_tensors(Tensor& multiplicand, Tensor& multiplier) {
  Tensor product =
      Tensor::from_xarray_(xt::eval(multiplicand.data_ * multiplier.data_));
  if (multiplicand.requires_grad || multiplier.requires_grad) {
    product.gradient_fn = new MulBackward(multiplicand, multiplier);
    product.gradient_fn->saved_tensors.insert(
        product.gradient_fn->saved_tensors.begin(),
        {multiplicand.save(), multiplier.save()});
    product.requires_grad = true;
  }
  return product;
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

MulBackward::MulBackward(Tensor& multiplicand, Tensor& multiplier) {
  if (multiplicand.requires_grad) {
    edges.push_back(
        autograd::Edge(MULTIPLICAND_INDEX, multiplicand.get_gradient_edge()));
  }
  if (multiplier.requires_grad) {
    edges.push_back(
        autograd::Edge(MULTIPLIER_INDEX, multiplier.get_gradient_edge()));
  }
}

std::vector<Tensor> MulBackward::operator()(Tensor output_grad) {
  auto multiplicand = saved_tensors[MULTIPLICAND_INDEX];
  auto multiplier = saved_tensors[MULTIPLIER_INDEX];

  xt::xarray<double> multiplier_grad_raw =
      multiplicand.data_ * output_grad.data_;
  xt::xarray<double> multiplicand_grad_raw =
      multiplier.data_ * output_grad.data_;

  auto multiplicand_grad =
      reduce_broadcast(multiplicand_grad_raw, multiplicand.data_.shape());
  auto multiplier_grad =
      reduce_broadcast(multiplier_grad_raw, multiplier.data_.shape());

  return {Tensor::from_xarray_(multiplicand_grad),
          Tensor::from_xarray_(multiplier_grad)};
}

}  // namespace ember
