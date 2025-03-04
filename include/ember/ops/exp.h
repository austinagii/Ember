#ifndef EMBER_OPS_EXP_H
#define EMBER_OPS_EXP_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

/**
 * Performs the exponent operation on each element in the tensor.
 *
 * i.e. $y_{i} = e^{x_i}$
 */
Tensor exp(Tensor& exponent);

/**
 * Represents the backwards pass of the exponent function.
 */
struct ExpBackward : public autograd::Node {
  ExpBackward(Tensor& exponent);
  virtual std::vector<Tensor> operator()(Tensor output_grad);
};

}  // namespace ember

#endif  // !EMBER_OPS_EXP_H
