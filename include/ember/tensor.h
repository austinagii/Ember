#ifndef EMBER_TENSOR_H
#define EMBER_TENSOR_H

#include <ember/autograd/node.h>
#include <ember/autograd/engine.h>
#include <ember/autograd/accumulator.h>
#include <ember/ops/add.h>
#include <ember/ops/sub.h>
#include <ember/ops/mul.h>
#include <ember/ops/div.h>
#include <ember/tensor_snapshot.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include <initializer_list>
#include <numeric>  
#include <functional>  
#include <vector>
#include <type_traits>

namespace ember {

/**
 * Tensor is the central resource of the system. It represents the core resources that are
 * manipulated and stored and act as inputs and outputs to the system. Computational graphs 
 * are built by performing operations on Tensors.
 *
 * Tensors are thin wrappers around a mathematical tensor (i.e. multidemnsional array) that 
 * provide additional capabilities for calculating gradients and storing gradient information
 * as well as hooking into the computational graph in which it participates.
 * 
 * This class corresponds to the `Variable` class in PyTorch's autograd, the difference in 
 * naming here stems from the fact that this naming in PyTorch in intended to be deprecated 
 * in favor of the current naming. 
 */
struct Tensor {

  // The multidemnsional array of data this tensor contains.
  xt::xarray<double> data_;
  // The gradient of this node w.r.t. the ancestor on which backward was called.
  Tensor* gradient = nullptr;
  // The function that will be used to pass the gradient from this tensor to it's parents.
  autograd::Node* gradient_fn = nullptr;
  // Accumulates a sum of gradients for this tensor if it is a leaf tensor.
  autograd::Node* gradient_accumulator = nullptr;

  Tensor();

  Tensor(double value);
  
  // 1D tensor constructor
  Tensor(std::initializer_list<double> values);

  // 2D tensor constructor
  Tensor(std::initializer_list<std::initializer_list<double>> values);

  // 3D tensor constructor
  Tensor(std::initializer_list<std::initializer_list<std::initializer_list<double>>> values);
  
  static Tensor from_xarray_(xt::xarray<double> data) {
    auto t = Tensor();
    t.data_ = data;
    return t;
  }

  static Tensor zeros_like(const Tensor& other) {
    return Tensor::from_xarray_(xt::zeros_like(other.data_));
  }

  /**
   * @brief Creates a new tensor with the same shape as the input tensor, but 
   * with all elements set to 1.
   */
  static Tensor ones_like(const Tensor& other);

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
  * Approximate equality means that the tensors both have the same shape and there
  * is element-wise equality within a given range.
  */
  bool equals_approx(const Tensor& other);

  /**
   * @brief Computes the gradient of this tensor w.r.t. the ancestor on which backward was called.
   */ 
  void backward();

  /**
   * @brief Returns the gradient edge for this tensor.
   */
  autograd::Node* get_gradient_edge();

  /**
   * @brief Saves the current state of this tensor.
   */
  TensorSnapshot save();

  /**
   * @brief Access a tensor element (const version)
   */
  template<typename... Args>
  double operator()(Args... args) const {
    return data_(args...);
  }

  /**
   * @brief Access a tensor element (mutable version)
   */
  template<typename... Args>
  double& operator()(Args... args) {
    return data_(args...);
  }

  /**
   * @brief Creates a new tensor with the specified shape, initialized to zeros.
   */
  static Tensor from_shape(std::initializer_list<size_t> shape);
  
  // Add the friend declaration inside the class
  friend struct TensorSnapshot;
}; // struct Tensor

} // namespace ember

#endif // EMBER_TENSOR_H
