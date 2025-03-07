#ifndef EMBER_TENSOR_H
#define EMBER_TENSOR_H

#include <ember/autograd/accumulator.h>
#include <ember/autograd/engine.h>
#include <ember/autograd/node.h>
#include <ember/ops/add.h>
#include <ember/ops/div.h>
#include <ember/ops/exp.h>
#include <ember/ops/matmul.h>
#include <ember/ops/mul.h>
#include <ember/ops/sub.h>

#include <ember/tensor_snapshot.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

template <typename T> using init_list = std::initializer_list<T>;

namespace ember {

/**
 * Tensor is the central resource of the Ember. It represents the core
 * resource that is created, manipulated and stored acting as inputs to and
 * outputs of various operations.
 *
 * Ember Tensors are thin wrappers around xtensor (multidimensional) arrays
 * providing additional scaffolding for calculating and storing gradients as
 * well as hooking into the computational graph.
 *
 * This class corresponds to the `Variable` class in PyTorch's autograd.
 */
struct Tensor {
public:
  // The multidimensional array of data this tensor contains.
  xt::xarray<double> data_;
  // The gradient of this node w.r.t. the ancestor on which backward was
  // called.
  Tensor* gradient = nullptr;

  // Default constructor
  Tensor();

  // Constructor with requires_grad flag
  explicit Tensor(bool requires_grad);

  /**
   * @brief Constructs a tensor with a single value.
   * @param value The value to initialize the tensor with
   * @param requires_grad If true, the tensor will track gradients for
   * autograd
   * @example
   *   Tensor t(1.0); // Creates a tensor with a single value 1.0
   */
  Tensor(double value, bool requires_grad = false);

  /**
   * @brief Constructs a 1-dimensional tensor from a list of values.
   * @param values The values to initialize the tensor with
   * @param requires_grad If true, the tensor will track gradients for
   * autograd
   * @example
   *   Tensor t({1.0, 2.0, 3.0}); // Creates a 1D tensor [1.0, 2.0, 3.0]
   */
  Tensor(init_list<double> values, bool requires_grad = false);

  /**
   * @brief Constructs a 2-dimensional tensor from a nested list of values.
   * @param values The values to initialize the tensor with
   * @param requires_grad If true, the tensor will track gradients for
   * autograd
   * @example
   *   Tensor t({
   *     {1.0, 2.0},
   *     {3.0, 4.0}
   *   }); // Creates a 2x2 tensor [[1.0, 2.0], [3.0, 4.0]]
   */
  Tensor(init_list<init_list<double>> values, bool requires_grad = false);

  /**
   * @brief Constructs a 3-dimensional tensor from a doubly-nested list of
   * values.
   * @param values The values to initialize the tensor with
   * @param requires_grad If true, the tensor will track gradients for
   * autograd
   * @example
   *   Tensor t({
   *     {{1.0, 2.0}, {3.0, 4.0}},
   *     {{5.0, 6.0}, {7.0, 8.0}}
   *   }); // Creates a 2x2x2 tensor
   */
  Tensor(init_list<init_list<init_list<double>>> values,
         bool requires_grad = false);

  /**
   * @brief Copy constructor that creates a deep copy of another tensor
   * @param other The tensor to copy from
   */
  Tensor(const Tensor& other);

  /**
   * @brief Sets whether this tensor requires gradients.
   * @param requires_grad If true, the tensor will track gradients for autograd
   * @return A reference to this tensor
   */
  Tensor& requires_grad(bool requires_grad);

  /**
   * @brief Gets whether this tensor requires gradients.
   * @return True if the tensor requires gradients, false otherwise
   */
  bool requires_grad() const;

  /**
   * @brief Gets the gradient function for this tensor.
   * @return Pointer to the gradient function node
   */
  autograd::Node* get_gradient_fn() const;

  /**
   * @brief Sets the gradient function for this tensor.
   * @param gradient_fn Pointer to the gradient function node
   */
  void set_gradient_fn(autograd::Node* gradient_fn);

  /**
   * @brief Access a tensor element (const version)
   */
  template <typename... Args> double operator()(Args... args) const {
    return data_(args...);
  }

  /**
   * @brief Access a tensor element (mutable version)
   */
  template <typename... Args> double& operator()(Args... args) {
    return data_(args...);
  }

  /**
   * @brief Computes gradients for all input tensors that created this tensor,
   * using the provided gradient as the starting point for backpropagation.
   * @param gradient The initial gradient to begin backpropagation with
   */
  void backward(const Tensor& gradient);

  /**
   * @brief Computes gradients for all input tensors that created this tensor,
   * starting with a gradient of ones.
   */
  void backward();

  /**
   * @brief Performs the matrix multiplication between this tensor and the one
   * specified.
   *
   * This is equivalent to `ember::matmul(this, other)`.
   */
  Tensor matmul(Tensor& other);

  /**
   * @brief Computes the exponential of each element in the tensor.
   * @return A new tensor with the exponential of each element
   */
  Tensor exp();

  /**
   * @brief Compares two tensors to determine if they are exactly equal.
   *
   * Exact equality means that the tensors both have the same shape and there
   * is element-wise equality.
   */
  friend bool operator==(const Tensor& left, const Tensor& right);

  /**
   * @brief Compares two tensors to determine if they are approximately equal.
   *
   * Approximate equality means that the tensors both have the same shape and
   * there is element-wise equality within a given range.
   */
  bool equals_approx(const Tensor& other);

  /**
   * @brief Saves the current state of this tensor.
   */
  TensorSnapshot save();

  /**
   * @brief Creates a tensor from an existing xarray.
   * @param data The xarray to create the tensor from
   * @return A new tensor containing the provided data
   */
  static Tensor from_xarray_(xt::xarray<double> data) {
    auto t = Tensor();
    t.data_ = data;
    return t;
  }

  /**
   * @brief Creates a new tensor with the specified shape, initialized to
   * zeros.
   */
  static Tensor from_shape(std::initializer_list<size_t> shape);

  /**
   * @brief Creates a new tensor with the same shape as the input tensor, but
   * with all elements set to 0.
   */
  static Tensor zeros_like(const Tensor& other) {
    return Tensor::from_xarray_(xt::zeros_like(other.data_));
  }

  /**
   * @brief Creates a new tensor with the same shape as the input tensor, but
   * with all elements set to 1.
   */
  static Tensor ones_like(const Tensor& other);

  /**
   * @brief Creates a new tensor of the specified shape with each element
   * being a random number sampled from a normal distribution with the
   * specified mean and standard deviation.
   *
   * @param shape The shape of the tensor to create
   * @param mean The mean of the normal distribution
   * @param std The standard deviation of the normal distribution
   * @return A tensor of the specified shape with values sampled from a
   * normal distribution
   * @example
   *   Tensor t = Tensor::randn({2, 2}, 0.0, 1.0); // Creates a 2x2 tensor
   *   with values sampled from a normal distribution with mean 0.0 and
   *   standard deviation 1.0
   */
  static Tensor randn(std::initializer_list<size_t> shape, double mean = 0.0,
                      double std = 1.0);

private:
  // The function that will be used to pass the gradient from this tensor to
  // its parents.
  autograd::Node* gradient_fn = nullptr;
  // Accumulates a sum of gradients for this tensor if it is a leaf tensor.
  autograd::Node* gradient_accumulator = nullptr;
  // Whether this tensor requires gradients to be computed and stored.
  bool requires_grad_ = false;

  friend struct TensorSnapshot;
};  // class Tensor

Tensor operator+(Tensor& augend, Tensor& addend);
Tensor operator+(Tensor& augend, Tensor&& addend);
Tensor operator+(Tensor&& augend, Tensor& addend);
Tensor operator+(Tensor&& augend, Tensor&& addend);

}  // namespace ember

#endif  // EMBER_TENSOR_H
