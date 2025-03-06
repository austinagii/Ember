#ifndef EMBER_OPS_UTILS_H
#define EMBER_OPS_UTILS_H

#include "xtensor/xarray.hpp"

// TODO: Consider replacing macro with factory functions
#define REGISTER_OP_BACKWARD(name, backward_fn)                                \
  struct name##Backward : public autograd::Node {                              \
    template <typename... Tensors>                                             \
    name##Backward(autograd::Context ctx, Tensors&... inputs)                  \
        : autograd::Node() {                                                   \
      this->ctx = ctx;                                                         \
      std::size_t input_ix = 0;                                                \
      auto add_input = [this, &input_ix](auto& tensor) {                       \
        if (tensor.requires_grad()) {                                          \
          add_next_edge(autograd::Edge(input_ix, tensor.get_gradient_fn()));   \
        }                                                                      \
        input_ix += 1;                                                         \
      };                                                                       \
      (add_input(inputs), ...);                                                \
    }                                                                          \
                                                                               \
    std::vector<Tensor> operator()(Tensor output_grad) override {              \
      return backward_fn(ctx, output_grad);                                    \
    }                                                                          \
  };

#define REGISTER_UNARY_OP(name, forward_fn, backward_fn)                       \
  REGISTER_OP_BACKWARD(name, backward_fn)                                      \
                                                                               \
  Tensor name(Tensor& input) {                                                 \
    autograd::Context ctx;                                                     \
    Tensor output = forward_fn(ctx, input);                                    \
    if (input.requires_grad()) {                                               \
      output.requires_grad(true);                                              \
      output.set_gradient_fn(new name##Backward(ctx, input));                  \
    }                                                                          \
    return output;                                                             \
  }

#define REGISTER_BINARY_OP(name, forward_fn, backward_fn)                      \
  REGISTER_OP_BACKWARD(name, backward_fn)                                      \
                                                                               \
  Tensor name(Tensor& input1, Tensor& input2) {                                \
    autograd::Context ctx;                                                     \
    Tensor output = forward_fn(ctx, input1, input2);                           \
    if (input1.requires_grad() || input2.requires_grad()) {                    \
      output.requires_grad(true);                                              \
      output.set_gradient_fn(new name##Backward(ctx, input1, input2));         \
    }                                                                          \
    return output;                                                             \
  }

namespace ember {

xt::xarray<double> reduce_broadcast(
    const xt::xarray<double>& source,
    const xt::xarray<double>::shape_type& target_shape);

}  // namespace ember

#endif  // EMBER_OPS_UTILS_H
