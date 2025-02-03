#ifndef EMBER_OPS_SUB_H
#define EMBER_OPS_SUB_H

#include <vector>

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

struct SubBackward;

Tensor operator-(Tensor &minuend, Tensor &subtrahend);
Tensor operator-(Tensor &&minuend, Tensor &subtrahend);
Tensor operator-(Tensor &minuend, Tensor &&subtrahend);
Tensor operator-(Tensor &&minuend, Tensor &&subtrahend);

struct SubBackward : public autograd::Node {
  SubBackward(Tensor &minuend, Tensor &subtrahend);
  std::vector<Tensor> operator()(Tensor output_grad) override;
};  // struct SubBackward

}  // namespace ember

#endif  // EMBER_OPS_SUB_H
