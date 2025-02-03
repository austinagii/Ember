#include <ember/ops/sub.h>
#include <ember/ops/utils.h>

namespace ember {

std::size_t MINUEND_INDEX = 0;
std::size_t SUBTRAHEND_INDEX = 1;

static Tensor subtract_tensors(Tensor& minuend, Tensor& subtrahend) {
  Tensor difference =
      Tensor::from_xarray_(xt::eval(minuend.data_ - subtrahend.data_));
  if (minuend.requires_grad || subtrahend.requires_grad) {
    difference.gradient_fn = new SubBackward(minuend, subtrahend);
    difference.gradient_fn->saved_tensors.insert(
        difference.gradient_fn->saved_tensors.begin(),
        {minuend.save(), subtrahend.save()});
    difference.requires_grad = true;
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
  if (minuend.requires_grad) {
    edges.push_back(autograd::Edge(MINUEND_INDEX, minuend.get_gradient_edge()));
  }
  if (subtrahend.requires_grad) {
    edges.push_back(
        autograd::Edge(SUBTRAHEND_INDEX, subtrahend.get_gradient_edge()));
  }
}

xt::xarray<float> calculate_local_sub_gradient(xt::xarray<float> input,
                                               xt::xarray<float> output_grad) {
  auto input_shape = input.shape();
  auto input_rank = input_shape.size();
  auto output_shape = output_grad.shape();
  auto output_rank = output_shape.size();

  // align the shapes on their trailing dimesinsions and pad the input dims if
  // needed
  std::vector<std::size_t> padded_input_shape(output_rank, 1);
  for (std::size_t i = 0; i < input_rank; ++i) {
    padded_input_shape[output_rank - 1 - i] = input_shape[input_rank - 1 - i];
  }

  auto input_grad = output_grad;  // This needs to be a copy.
  for (int i = output_rank - 1; i >= 0; i--) {
    if (padded_input_shape[i] != output_shape[i]) {
      // we must reduce along this dimension.
      input_grad = xt::sum(input_grad, i);
    }
  }
  return input_grad;
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
