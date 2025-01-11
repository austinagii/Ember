#include <ember/ops/div.h>

namespace ember {

static Tensor divide_tensors(Tensor& dividend, Tensor& divisor) {
    auto quotient = Tensor(dividend.value / divisor.value);
    quotient.gradient_fn = new DivBackward(dividend, divisor);
    return quotient;
}

Tensor operator/(Tensor& dividend, Tensor& divisor) {
    return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor&& dividend, Tensor& divisor) {
    return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor& dividend, Tensor&& divisor) {
    return divide_tensors(dividend, divisor);
}

Tensor operator/(Tensor&& dividend, Tensor&& divisor) {
    return divide_tensors(dividend, divisor);
}

DivBackward::DivBackward(Tensor& dividend, Tensor& divisor) {
    saved_tensors.push_back(dividend.save());
    saved_tensors.push_back(divisor.save());

    next_fns.push_back(dividend.get_gradient_edge());
    next_fns.push_back(divisor.get_gradient_edge());
}

std::vector<Tensor> DivBackward::operator()(Tensor output_grad) {
    auto dividend = saved_tensors[0].value;
    auto divisor = saved_tensors[1].value;

    auto dividend_grad = Tensor(output_grad.value / divisor);
    auto divisor_grad = Tensor(output_grad.value * (-dividend / (divisor * divisor)));
    return {dividend_grad, divisor_grad};
}

}
