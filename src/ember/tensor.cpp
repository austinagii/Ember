#include <ember/autograd/node.h>  // Can this be removed?
#include <ember/tensor.h>

#include <xtensor/xrandom.hpp>

#include <functional>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ember {

Tensor::Tensor() {}

Tensor::Tensor(double value, bool requires_grad)
    : data_({value}), requires_grad(requires_grad) {}

Tensor::Tensor(init_list<double> values, bool requires_grad)
    : data_(values), requires_grad(requires_grad) {}

Tensor::Tensor(init_list<init_list<double>> values, bool requires_grad)
    : data_(values), requires_grad(requires_grad) {}

Tensor::Tensor(init_list<init_list<init_list<double>>> values,
               bool requires_grad)
    : data_(values), requires_grad(requires_grad) {}

Tensor::Tensor(const Tensor& other)
    : data_(other.data_), gradient_fn(other.gradient_fn),
      gradient_accumulator(other.gradient_accumulator),
      requires_grad(other.requires_grad) {
  if (other.gradient != nullptr) {
    gradient = new Tensor(*other.gradient);
  }
}

void Tensor::backward(const Tensor& gradient) {
  autograd::Engine engine;
  engine.backward(this->gradient_fn, gradient);
}

void Tensor::backward() { backward(Tensor::ones_like(*this)); }

autograd::Node* Tensor::get_gradient_edge() {
  if (gradient_fn != nullptr) {
    return gradient_fn;
  }

  if (gradient_accumulator == nullptr) {
    gradient_accumulator = new autograd::Accumulator(this);
  }
  return gradient_accumulator;
}

TensorSnapshot Tensor::save() { return TensorSnapshot(this); }

Tensor Tensor::matmul(Tensor& other) {
  return ember::matmul(*this, other);  
}

Tensor Tensor::exp() {
  return ember::exp(*this);
}

bool operator==(const Tensor& left, const Tensor& right) {
  return xt::all(xt::equal(left.data_, right.data_));
}

bool Tensor::equals_approx(const Tensor& other) {
  return xt::allclose(this->data_, other.data_);
}

Tensor Tensor::ones_like(const Tensor& other) {
  return Tensor::from_xarray_(xt::ones_like(other.data_));
}

Tensor Tensor::from_shape(std::initializer_list<size_t> shape) {
  return Tensor::from_xarray_(xt::xarray<double>::from_shape(shape));
}

Tensor Tensor::randn(std::initializer_list<size_t> shape, double mean,
                     double std) {
  return Tensor::from_xarray_(xt::random::randn<double>(shape, mean, std));
}

}  // namespace ember
