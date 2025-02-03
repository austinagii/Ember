#ifndef EMBER_OPS_DIV_H
#define EMBER_OPS_DIV_H

#include <vector>

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

struct DivBackward;

/**
 * Divides one tensor by another and returns a new tensor representing the
 * quotient.
 */
Tensor operator/(Tensor &dividend, Tensor &divisor);
Tensor operator/(Tensor &&dividend, Tensor &divisor);
Tensor operator/(Tensor &dividend, Tensor &&divisor);
Tensor operator/(Tensor &&dividend, Tensor &&divisor);

struct DivBackward : public autograd::Node {
  DivBackward(Tensor &dividend, Tensor &divisor);
  std::vector<Tensor> operator()(Tensor output_grad) override;
};  // struct DivBackward

}  // namespace ember

#endif  // EMBER_OPS_DIV_H
