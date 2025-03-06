#include <ember/tensor.h>

#include <ember/autograd/node.h>
#include <xtensor/xrandom.hpp>

#include <functional>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ember {

Tensor::Tensor() {}

Tensor::Tensor(bool requires_grad) { this->requires_grad(requires_grad); }

Tensor::Tensor(double value, bool requires_grad) : Tensor(requires_grad) {
  data_ = xt::xarray<double>({value});
}

Tensor::Tensor(init_list<double> values, bool requires_grad)
    : Tensor(requires_grad) {
  data_ = xt::xarray<double>(values);
}

Tensor::Tensor(init_list<init_list<double>> values, bool requires_grad)
    : Tensor(requires_grad) {
  data_ = xt::xarray<double>(values);
}

Tensor::Tensor(init_list<init_list<init_list<double>>> values,
               bool requires_grad)
    : Tensor(requires_grad) {
  data_ = xt::xarray<double>(values);
}

Tensor::Tensor(const Tensor& other)
    : data_(other.data_), gradient_fn(other.gradient_fn),
      gradient_accumulator(other.gradient_accumulator) {
  requires_grad_ = other.requires_grad();
  if (other.gradient != nullptr) {
    gradient = new Tensor(*other.gradient);
  }
}

Tensor& Tensor::requires_grad(bool requires_grad) {
  requires_grad_ = requires_grad;
  if (requires_grad_ && gradient_accumulator == nullptr) {
    gradient_accumulator = new autograd::Accumulator(this);
  }
  return *this;
}

bool Tensor::requires_grad() const { return requires_grad_; }

autograd::Node* Tensor::get_gradient_fn() const {
  if (gradient_fn == nullptr) {
    return gradient_accumulator;
  }
  return gradient_fn;
}

void Tensor::set_gradient_fn(autograd::Node* gradient_fn) {
  this->gradient_fn = gradient_fn;
}

void Tensor::backward(const Tensor& gradient) {
  if (gradient_fn == nullptr) {
    throw std::runtime_error(
        "backward called on a tensor that has no gradient function");
  }
  autograd::Engine engine;
  engine.backward(this->gradient_fn, gradient);
}

void Tensor::backward() { backward(Tensor::ones_like(*this)); }

Tensor Tensor::matmul(Tensor& other) { return ember::matmul(*this, other); }

Tensor Tensor::exp() { return ember::exp(*this); }

bool operator==(const Tensor& left, const Tensor& right) {
  return xt::all(xt::equal(left.data_, right.data_));
}

bool Tensor::equals_approx(const Tensor& other) {
  return xt::allclose(this->data_, other.data_);
}

TensorSnapshot Tensor::save() { return TensorSnapshot(this); }

Tensor Tensor::from_shape(std::initializer_list<size_t> shape) {
  return Tensor::from_xarray_(xt::xarray<double>::from_shape(shape));
}

Tensor Tensor::ones_like(const Tensor& other) {
  return Tensor::from_xarray_(xt::ones_like(other.data_));
}

Tensor Tensor::randn(std::initializer_list<size_t> shape, double mean,
                     double std) {
  return Tensor::from_xarray_(xt::random::randn<double>(shape, mean, std));
}

}  // namespace ember
