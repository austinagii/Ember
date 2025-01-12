#ifndef MUL_H
#define MUL_H

#include <vector>

#include <ember/tensor.h>
#include <ember/autograd/node.h>

namespace ember {

struct MulBackward;

/**
 * Multiply two tensors and return a new tensor representing the product.
 */
Tensor operator*(Tensor& multiplicand, Tensor& multiplier);
Tensor operator*(Tensor&& multiplicand, Tensor& multiplier);
Tensor operator*(Tensor& multiplicand, Tensor&& multiplier);
Tensor operator*(Tensor&& multiplicand, Tensor&& multiplier);

struct MulBackward: public autograd::Node {
    MulBackward(Tensor& multiplicand, Tensor& multiplier);
    std::vector<Tensor> operator()(Tensor output_grad) override;
}; // struct MulBackward

} // namespace ember

#endif // MUL_H 
