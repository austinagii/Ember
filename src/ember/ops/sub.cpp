#include <ember/ops/sub.h>
#include <ember/ops/utils.h>

namespace ember {

std::size_t MINUEND_INDEX = 0;
std::size_t SUBTRAHEND_INDEX = 1;

static Tensor subtract_tensors(Tensor& minuend, Tensor& subtrahend) {
  Tensor difference =
      Tensor::from_xarray_(xt::eval(minuend.data_ - subtrahend.data_));
  if (minuend.requires_grad() || subtrahend.requires_grad()) {
    difference.set_gradient_fn(new SubBackward(minuend, subtrahend));
    difference.get_gradient_fn()->saved_tensors.insert(
        difference.get_gradient_fn()->saved_tensors.begin(),
        {minuend.save(), subtrahend.save()});
    difference.requires_grad(true);
  }
  return difference;
}

Tensor operator-(Tensor& minuend, Tensor& subtrahend) {
  return subtract_tensors(minuend, subtrahend);
}

Tensor operator-(Tensor&& minuend, Tensor& subtrahend) {
  return subtract_tensors(minuend, subtrahend);
}

Tensor operator-(Tensor& minuend, Tensor&& subtrahend) {
  return subtract_tensors(minuend, subtrahend);
}

Tensor operator-(Tensor&& minuend, Tensor&& subtrahend) {
  return subtract_tensors(minuend, subtrahend);
}

SubBackward::SubBackward(Tensor& minuend, Tensor& subtrahend) {
  if (minuend.requires_grad()) {
    edges.push_back(autograd::Edge(MINUEND_INDEX, minuend.get_gradient_fn()));
  }
  if (subtrahend.requires_grad()) {
    edges.push_back(
        autograd::Edge(SUBTRAHEND_INDEX, subtrahend.get_gradient_fn()));
  }
}

std::vector<Tensor> SubBackward::operator()(Tensor output_grad) {
  auto minuend = saved_tensors[MINUEND_INDEX];
  auto subtrahend = saved_tensors[SUBTRAHEND_INDEX];

  xt::xarray<double> minuend_grad_broadcasted = output_grad.data_;
  xt::xarray<double> subtrahend_grad_broadcasted = -1 * output_grad.data_;

  xt::xarray<double> minuend_grad =
      reduce_broadcast(minuend_grad_broadcasted, minuend.data_.shape());
  xt::xarray<double> subtrahend_grad =
      reduce_broadcast(subtrahend_grad_broadcasted, subtrahend.data_.shape());

  return {Tensor::from_xarray_(minuend_grad),
          Tensor::from_xarray_(subtrahend_grad)};
}

}  // namespace ember
