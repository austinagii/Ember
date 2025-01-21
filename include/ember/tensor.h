#ifndef EMBER_TENSOR_H
#define EMBER_TENSOR_H

#include <ember/autograd/node.h>
#include <ember/autograd/engine.h>
#include <ember/autograd/accumulator.h>
#include <ember/ops/add.h>
#include <ember/ops/sub.h>
#include <ember/ops/mul.h>
// #include <ember/ops/div.h>
#include <ember/tensor_snapshot.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include <initializer_list>
#include <numeric>  
#include <functional>  
#include <vector>
#include <type_traits>

namespace ember {

// Forward declare TensorSnapshot
struct TensorSnapshot;

/**
* This class corresponds to the `Variable` class in PyTorch's autograd, the
* difference in naming here stems from the fact that this naming in PyTorch 
* in intended to be deprecated in favor of the current naming. 
*/
struct Tensor {
  xt::xarray<float> data;
  // The gradient of this node w.r.t. the ancestor on which backward was called.
  Tensor* gradient = nullptr;
  // The function that will be used to pass the gradient from this tensor to it's parents.
  autograd::Node* gradient_fn = nullptr;
  // Accumulates a sum of gradients for this tensor if it is a leaf tensor.
  autograd::Node* gradient_accumulator = nullptr;

  Tensor();
  Tensor(xt::xarray<float> data);
  
  void backward();
  autograd::Node* get_gradient_edge();
  TensorSnapshot save();
}; // struct Tensor

} // namespace ember

#endif // EMBER_TENSOR_H
