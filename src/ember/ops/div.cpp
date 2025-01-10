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

DivBackward::DivBackward(Tensor& dividend, Tensor& divisor): 
    dividend(dividend), 
    divisor(divisor) 
{
    next_fns.push_back(dividend.get_gradient_edge());
    next_fns.push_back(divisor.get_gradient_edge());
}

std::vector<Tensor> DivBackward::operator()(Tensor output_grad) {
    auto dividend_grad = Tensor(output_grad.value / divisor.value);
    auto divisor_grad = Tensor(output_grad.value * (-dividend.value) / (divisor.value * divisor.value));
    return {dividend_grad, divisor_grad};
}

}
