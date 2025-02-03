#ifndef EMBER_OPS_ADD_H
#define EMBER_OPS_ADD_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <xtensor/xarray.hpp>

#include <vector>

namespace ember {

struct AddBackward;

/**
 * @brief Add two tensors and return a new tensor representing the sum.
 */
Tensor operator+(Tensor& augend, Tensor& addend);
Tensor operator+(Tensor& augend, Tensor&& addend);
Tensor operator+(Tensor&& augend, Tensor& addend);
Tensor operator+(Tensor&& augend, Tensor&& addend);

/**
 * @brief The node representing the backward pass for the addition operation.
 */
struct AddBackward : public autograd::Node {
  AddBackward(Tensor& augend, Tensor& addend);

  /**
   * @brief Compute the gradients of both inputs of an addition operation (a+b)
   *
   * Given that c = a + b and the partial derivate of c w.r.t some output o
   * (∂c/∂o) is `output_grad`, then the partial derivatives of a and b are as
   * follows:
   *
   * - ∂a/∂o = `output_grad` * 1
   * - ∂b/∂o = `output_grad` * 1
   *
   * @param output_grad The gradient of the output of the addition operation.
   * @return A vector containing the gradients of the inputs of the addition
   * operation.
   */
  virtual std::vector<Tensor> operator()(Tensor output_grad);
};

}  // namespace ember

#endif  // EMBER_OPS_ADD_H
