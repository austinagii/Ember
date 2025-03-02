#ifndef EMBER_OPS_MATMUL_H
#define EMBER_OPS_MATMUL_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

/** 
 * Performs matrix multiplication of a and b. 
 */
Tensor matmul(Tensor& a, Tensor& b);

/**
 * @brief Represents the function that calculates and propagates the output 
 * gradient of the dot operation to the input tensors.
 */
struct MatmulBackward: public autograd::Node {
  MatmulBackward(Tensor& a, Tensor& b);
  virtual std::vector<Tensor> operator()(Tensor output_grad);
};

}  // namespace ember

#endif  // EMBER_OPS_MATMUL_H
