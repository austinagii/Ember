#ifndef SUB_H
#define SUB_H

#include <vector>

#include <ember/tensor.h>
#include <ember/autograd/node.h>

namespace ember {

struct SubBackward;

Tensor operator-(Tensor& minuend, Tensor& subtrahend);
Tensor operator-(Tensor&& minuend, Tensor& subtrahend);
Tensor operator-(Tensor& minuend, Tensor&& subtrahend);
Tensor operator-(Tensor&& minuend, Tensor&& subtrahend);

struct SubBackward: public autograd::Node {
    SubBackward(Tensor& minuend, Tensor& subtrahend);
    std::vector<Tensor> operator()(Tensor output_grad) override;
}; // struct SubBackward

} // namespace ember

#endif // SUB_H