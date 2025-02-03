#ifndef EMBER_OPS_MUL_H
#define EMBER_OPS_MUL_H

#include <vector>

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

struct MulBackward;

/**
 * @brief Multiply two tensors and return a new tensor representing the product.
 */
Tensor operator*(Tensor& multiplicand, Tensor& multiplier);
Tensor operator*(Tensor&& multiplicand, Tensor& multiplier);
Tensor operator*(Tensor& multiplicand, Tensor&& multiplier);
Tensor operator*(Tensor&& multiplicand, Tensor&& multiplier);

/**
 * @brief The node representing the backward pass for the multiplication
 * operation.
 */
struct MulBackward : public autograd::Node {
  MulBackward(Tensor& multiplicand, Tensor& multiplier);

  /**
   * @brief Compute the gradients of both inputs of a multiplication operation
   * (a*b)
   *
   * Given that c = a * b and the partial derivate of c w.r.t some output o
   * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
   * follows:
   *
   * - ∂a/∂o = `output_grad` * b
   * - ∂b/∂o = `output_grad` * a
   *
   * @param output_grad The gradient of the output of the multiplication
   * operation.
   * @return A vector containing the gradients of the inputs of the
   * multiplication operation.
   */
  std::vector<Tensor> operator()(Tensor output_grad) override;
};  // struct MulBackward

}  // namespace ember

#endif  // MUL_H
