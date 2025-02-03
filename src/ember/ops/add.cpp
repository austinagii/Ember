#include <ember/ops/add.h>
#include <ember/tensor.h>
#include <ember/autograd/node.h>

#include <xtensor/xbroadcast.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>

namespace ember {

std::size_t AUGEND_INDEX = 0;
std::size_t ADDEND_INDEX = 1;

static Tensor add_tensors(Tensor& augend, Tensor& addend) {
  Tensor sum = Tensor::from_xarray_(xt::eval(augend.data_ + addend.data_));

  if (augend.requires_grad || addend.requires_grad) {
    sum.gradient_fn = new AddBackward(augend, addend);
    sum.gradient_fn->saved_tensors.insert(sum.gradient_fn->saved_tensors.begin(), 
                                        {augend.save(), addend.save()});
    sum.requires_grad = true;
  }
  return sum;
}

Tensor operator+(Tensor& augend, Tensor& addend) {
  return add_tensors(augend, addend);
}

Tensor operator+(Tensor& augend, Tensor&& addend) {
  return add_tensors(augend, addend);
}

Tensor operator+(Tensor&& augend, Tensor& addend) {
  return add_tensors(augend, addend);
}

Tensor operator+(Tensor&& augend, Tensor&& addend) {
  return add_tensors(augend, addend);
}

AddBackward::AddBackward(Tensor& augend, Tensor& addend) {
  if (augend.requires_grad) {
    edges.push_back(autograd::Edge(0, augend.get_gradient_edge()));
  }
  if (addend.requires_grad) {
    edges.push_back(autograd::Edge(1, addend.get_gradient_edge()));
  }
}

/**
 * Caclulate the partial derivative of `input` w.r.t. `output` for an addition operation.
 *
 * This function works by first determining the broadcast shape for the input tensor. It then
 * iterates over each dimension, calculating the scaling factor required for the input in that
 * dimension. These scaling factors are collected into a cumulative product, which is then
 * multiplied by the identity matrix with the same shape as the input to produce `input`'s 
 * gradient tensor. 
 */
xt::xarray<float> calculate_local_add_gradient(xt::xarray<float> input, xt::xarray<float> output_grad) {
  auto input_shape = input.shape();
  auto input_rank = input_shape.size();
  auto output_shape = output_grad.shape();
  auto output_rank = output_shape.size();

  // align the shapes on their trailing dimesinsions and pad the input dims if needed
  std::vector<std::size_t> padded_input_shape(output_rank, 1);
  for (std::size_t i = 0; i < input_rank; ++i) {
    padded_input_shape[output_rank - 1 - i] = input_shape[input_rank - 1 - i];
  }

  auto input_grad = output_grad;  // This needs to be a copy.
  for (int i = output_rank - 1; i >= 0; i--) {
    if (padded_input_shape[i] != output_shape[i]) {
      // we must collapse along this dimension.
      input_grad = xt::sum(input_grad, i);
    }
  }
  return input_grad;
}

std::vector<Tensor> AddBackward::operator()(Tensor output_grad) {
  auto augend = saved_tensors[AUGEND_INDEX];
  auto addend = saved_tensors[ADDEND_INDEX];

  auto augend_gradient = calculate_local_add_gradient(augend.data_, output_grad.data_);
  auto addend_gradient = calculate_local_add_gradient(addend.data_, output_grad.data_);
  return {Tensor::from_xarray_(augend_gradient), Tensor::from_xarray_(addend_gradient)};
}

}
