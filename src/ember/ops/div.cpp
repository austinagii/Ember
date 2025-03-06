#include <ember/ops/div.h>
#include <ember/ops/utils.h>

namespace ember {

std::size_t DIVIDEND_INDEX = 0;
std::size_t DIVISOR_INDEX = 1;

static Tensor divide_tensors(Tensor& dividend, Tensor& divisor) {
  if (xt::any(xt::equal(divisor.data_, 0.0))) {
    throw std::runtime_error("Division by zero is not allowed");
  }
  Tensor quotient =
      Tensor::from_xarray_(xt::eval(dividend.data_ / divisor.data_));

  if (dividend.requires_grad() || divisor.requires_grad()) {
    quotient.set_gradient_fn(new DivBackward(dividend, divisor));
    quotient.get_gradient_fn()->saved_tensors.insert(
        quotient.get_gradient_fn()->saved_tensors.begin(),
        {dividend.save(), divisor.save()});
    quotient.requires_grad(true);
  }
  return quotient;
}

Tensor operator/(Tensor& dividend, Tensor& divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor&& dividend, Tensor& divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor& dividend, Tensor&& divisor) {
  return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor&& dividend, Tensor&& divisor) {
  return divide_tensors(dividend, divisor);
}

DivBackward::DivBackward(Tensor& dividend, Tensor& divisor) {
  if (dividend.requires_grad()) {
    edges.push_back(autograd::Edge(DIVIDEND_INDEX, dividend.get_gradient_fn()));
  }
  if (divisor.requires_grad()) {
    edges.push_back(autograd::Edge(DIVISOR_INDEX, divisor.get_gradient_fn()));
  }
}

std::vector<Tensor> DivBackward::operator()(Tensor output_grad) {
  auto dividend = saved_tensors[DIVIDEND_INDEX];
  auto divisor = saved_tensors[DIVISOR_INDEX];

  auto dividend_grad_raw = xt::eval(output_grad.data_ / divisor.data_);
  auto divisor_grad_raw = xt::eval(
      output_grad.data_ * (-dividend.data_ / (divisor.data_ * divisor.data_)));

  auto dividend_grad =
      reduce_broadcast(dividend_grad_raw, dividend.data_.shape());
  auto divisor_grad = reduce_broadcast(divisor_grad_raw, divisor.data_.shape());

  return {Tensor::from_xarray_(dividend_grad),
          Tensor::from_xarray_(divisor_grad)};
}

}  // namespace ember
