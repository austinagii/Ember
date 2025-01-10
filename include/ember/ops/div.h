#ifndef DIV_H
#define DIV_H

#include <vector>

#include <ember/tensor.h>
#include <ember/autograd/node.h>

namespace ember {

struct DivBackward;

/**
 * Divides one tensor by another and returns a new tensor representing the quotient.
 */
Tensor operator/(Tensor& dividend, Tensor& divisor);
Tensor operator/(Tensor&& dividend, Tensor& divisor);
Tensor operator/(Tensor& dividend, Tensor&& divisor);
Tensor operator/(Tensor&& dividend, Tensor&& divisor);

struct DivBackward: public autograd::Node {
    DivBackward(Tensor& dividend, Tensor& divisor);
    std::vector<Tensor> operator()(Tensor output_grad) override;

private:
    Tensor& dividend;
    Tensor& divisor;
}; // struct DivBackward

} // namespace ember

#endif // DIV_H
