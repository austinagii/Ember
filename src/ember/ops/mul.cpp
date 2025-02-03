#include <ember/ops/mul.h>

namespace ember {

std::size_t MULTIPLICAND_INDEX = 0;
std::size_t MULTIPLIER_INDEX = 1;

static Tensor multiply_tensors(Tensor &multiplicand, Tensor &multiplier) {
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

Tensor operator*(Tensor &multiplicand, Tensor &multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

Tensor operator*(Tensor &&multiplicand, Tensor &multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

Tensor operator*(Tensor &multiplicand, Tensor &&multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

Tensor operator*(Tensor &&multiplicand, Tensor &&multiplier) {
  return multiply_tensors(multiplicand, multiplier);
}

MulBackward::MulBackward(Tensor &multiplicand, Tensor &multiplier) {
  if (multiplicand.requires_grad) {
    edges.push_back(
        autograd::Edge(MULTIPLICAND_INDEX, multiplicand.get_gradient_edge()));
  }
  if (multiplier.requires_grad) {
    edges.push_back(
        autograd::Edge(MULTIPLIER_INDEX, multiplier.get_gradient_edge()));
  }
}

xt::xarray<float> calculate_local_mul_gradient(xt::xarray<float> input,
                                               xt::xarray<float> other,
                                               xt::xarray<float> output_grad) {
  auto input_shape = input.shape();
  auto input_rank = input_shape.size();
  auto output_shape = output_grad.shape();
  auto output_rank = output_shape.size();

  // Align shapes on trailing dimensions and pad input dims if needed
  std::vector<std::size_t> padded_input_shape(output_rank, 1);
  for (std::size_t i = 0; i < input_rank; ++i) {
    padded_input_shape[output_rank - 1 - i] = input_shape[input_rank - 1 - i];
  }

  // For multiplication, gradient is: output_grad * other
  auto input_grad = xt::eval(output_grad * other);

  // Reduce along broadcast dimensions
  for (int i = output_rank - 1; i >= 0; i--) {
    if (padded_input_shape[i] != output_shape[i]) {
      input_grad = xt::sum(input_grad, i);
    }
  }
  return input_grad;
}

std::vector<Tensor> MulBackward::operator()(Tensor output_grad) {
  auto multiplicand = saved_tensors[MULTIPLICAND_INDEX];
  auto multiplier = saved_tensors[MULTIPLIER_INDEX];

  // For multiplication, partial derivatives are:
  // ∂(a*b)/∂a = b * output_grad
  // ∂(a*b)/∂b = a * output_grad
  auto multiplicand_grad = calculate_local_mul_gradient(
      multiplicand.data_, multiplier.data_, output_grad.data_);
  auto multiplier_grad = calculate_local_mul_gradient(
      multiplier.data_, multiplicand.data_, output_grad.data_);

  return {Tensor::from_xarray_(multiplicand_grad),
          Tensor::from_xarray_(multiplier_grad)};
}

}  // namespace ember
