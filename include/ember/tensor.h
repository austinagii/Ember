#ifndef TENSOR_H
#define TENSOR_H

#include <ember/autograd/node.h>
#include <ember/autograd/engine.h>
#include <ember/autograd/accumulator.h>

// TODO: Include operation headers here to allow for including tensor w/ operations.

namespace ember {

/**
* This class corresponds to the `Variable` class in PyTorch's autograd, the
* difference in naming here stems from the fact that this naming in PyTorch 
* in intended to be deprecated in favor of the current naming. 
*/
struct Tensor {
  // The current value of this tensor. Only scalar values are currently supported. 
  float value;  
  // The gradient of this node w.r.t. the ancestor on which backward was called.
  Tensor* gradient = nullptr;
  // The function that will be used to pass the gradient from this tensor to it's parents.
  autograd::Node* gradient_fn = nullptr;
  // Accumulates a sum of gradients for this tensor if it is a leaf tensor.
  autograd::Node* gradient_accumulator = nullptr;

  Tensor();
  Tensor(float value);
  void backward(); 
  autograd::Node* get_gradient_edge();

}; // struct Tensor

} // namespace ember

#endif // TENSOR_H
