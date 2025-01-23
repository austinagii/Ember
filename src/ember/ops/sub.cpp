#include <ember/ops/sub.h>

namespace ember {

std::size_t MINUEND_INDEX = 0;
std::size_t SUBTRAHEND_INDEX = 1;

static Tensor subtract_tensors(Tensor& minuend, Tensor& subtrahend) {
  Tensor difference = Tensor::from_xarray(xt::eval(minuend.data - subtrahend.data));
  auto gradient_fn = new SubBackward(minuend, subtrahend);
  gradient_fn->saved_tensors.insert(gradient_fn->saved_tensors.begin(), 
                                  {minuend.save(), subtrahend.save()});
  difference.gradient_fn = gradient_fn;
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
    edges.push_back(autograd::Edge(0, minuend.get_gradient_edge()));
    edges.push_back(autograd::Edge(1, subtrahend.get_gradient_edge()));
}

xt::xarray<float> calculate_local_sub_gradient(xt::xarray<float> input, xt::xarray<float> output_grad) {
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
      // we must reduce along this dimension.
      input_grad = xt::sum(input_grad, i);
    }
  }
  return input_grad;
}

std::vector<Tensor> SubBackward::operator()(Tensor output_grad) {
  auto minuend = saved_tensors[MINUEND_INDEX];
  auto subtrahend = saved_tensors[SUBTRAHEND_INDEX];

  auto minuend_grad = Tensor(calculate_local_sub_gradient(minuend.data, output_grad.data));
  auto subtrahend_grad = Tensor(-1 * calculate_local_sub_gradient(subtrahend.data, output_grad.data));
  return {minuend_grad, subtrahend_grad};
}

} // namespace ember
