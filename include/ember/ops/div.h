#ifndef EMBER_OPS_DIV_H
#define EMBER_OPS_DIV_H

#include <vector>

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

struct DivBackward;

/**
 * @brief Divide two tensors and return a new tensor representing the quotient.
 */
Tensor operator/(Tensor& dividend, Tensor& divisor);
Tensor operator/(Tensor&& dividend, Tensor& divisor);
Tensor operator/(Tensor& dividend, Tensor&& divisor);
Tensor operator/(Tensor&& dividend, Tensor&& divisor);

/**
 * @brief The node representing the backward pass for the division operation.
 */
struct DivBackward : public autograd::Node {
  DivBackward(Tensor& dividend, Tensor& divisor);

  /**
   * @brief Compute the gradients of both inputs of a division operation (a/b)
   *
   * Given that c = a / b and the partial derivate of c w.r.t some output o
   * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
   * follows:
   *
   * - ∂a/∂o = `output_grad` / b
   * - ∂b/∂o = -`output_grad` * a / (b * b)
   *
   * @param output_grad The gradient of the output of the division operation.
   * @return A vector containing the gradients of the inputs of the division
   * operation.
   */
  std::vector<Tensor> operator()(Tensor output_grad) override;
};  // struct DivBackward

}  // namespace ember

#endif  // EMBER_OPS_DIV_H
