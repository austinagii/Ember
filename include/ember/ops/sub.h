#ifndef EMBER_OPS_SUB_H
#define EMBER_OPS_SUB_H

#include <vector>

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

struct SubBackward;

Tensor operator-(Tensor& minuend, Tensor& subtrahend);
Tensor operator-(Tensor&& minuend, Tensor& subtrahend);
Tensor operator-(Tensor& minuend, Tensor&& subtrahend);
Tensor operator-(Tensor&& minuend, Tensor&& subtrahend);

/**
 * @brief The node representing the backward pass for the subtraction operation.
 */
struct SubBackward : public autograd::Node {
  SubBackward(Tensor& minuend, Tensor& subtrahend);

  /**
   * @brief Compute the gradients of both inputs of a subtraction operation (a-b)
   *
   * Given that c = a - b and the partial derivate of c w.r.t some output o
   * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
   * follows:
   *
   * - ∂a/∂o = `output_grad` * 1
   * - ∂b/∂o = `output_grad` * -1
   *
   * @param output_grad The gradient of the output of the subtraction operation.
   * @return A vector containing the gradients of the inputs of the subtraction
   * operation.
   */
  std::vector<Tensor> operator()(Tensor output_grad) override;
};  // struct SubBackward

}  // namespace ember

#endif  // EMBER_OPS_SUB_H
