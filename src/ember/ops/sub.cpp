#include <ember/ops/sub.h>

namespace ember {

static Tensor subtract_tensors(Tensor& minuend, Tensor& subtrahend) {
    auto difference = Tensor(minuend.value - subtrahend.value);
    difference.gradient_fn = new SubBackward(minuend, subtrahend);
    return difference;
}

Tensor operator-(Tensor& minuend, Tensor& subtrahend) {
    return subtract_tensors(minuend, subtrahend);
}

Tensor operator-(Tensor&& minuend, Tensor& subtrahend) {
    return subtract_tensors(minuend, subtrahend);
}

Tensor operator-(Tensor& minuend, Tensor&& subtrahend) {
    return subtract_tensors(minuend, subtrahend);
}

Tensor operator-(Tensor&& minuend, Tensor&& subtrahend) {
    return subtract_tensors(minuend, subtrahend);
}

SubBackward::SubBackward(Tensor& minuend, Tensor& subtrahend) {
    next_fns.push_back(minuend.get_gradient_edge());
    next_fns.push_back(subtrahend.get_gradient_edge());
}

std::vector<Tensor> SubBackward::operator()(Tensor output_grad) {
    auto minuend_grad = Tensor(output_grad.value);
    auto subtrahend_grad = Tensor(-output_grad.value);
    return {minuend_grad, subtrahend_grad};
}

} // namespace ember