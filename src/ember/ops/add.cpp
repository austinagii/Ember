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

static Tensor add_tensors(Tensor& augend, Tensor& addend) {
  Tensor sum = Tensor::from_xarray_(xt::eval(augend.data_ + addend.data_));

  if (augend.requires_grad || addend.requires_grad) {
    sum.gradient_fn = new AddBackward(augend, addend);
    sum.gradient_fn->saved_tensors.insert(
        sum.gradient_fn->saved_tensors.begin(), {augend.save(), addend.save()});
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
    edges.push_back(autograd::Edge(AUGEND_INDEX, augend.get_gradient_edge()));
  }
  if (addend.requires_grad) {
    edges.push_back(autograd::Edge(ADDEND_INDEX, addend.get_gradient_edge()));
  }
}

std::vector<Tensor> AddBackward::operator()(Tensor output_grad) {
  auto augend = saved_tensors[AUGEND_INDEX];
  auto addend = saved_tensors[ADDEND_INDEX];

  auto augend_gradient =
      reduce_broadcast(output_grad.data_, augend.data_.shape());
  auto addend_gradient =
      reduce_broadcast(output_grad.data_, addend.data_.shape());
  return {Tensor::from_xarray_(augend_gradient),
          Tensor::from_xarray_(addend_gradient)};
}

}  // namespace ember
