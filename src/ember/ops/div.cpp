#include <ember/ops/div.h>

namespace ember {

std::size_t DIVIDEND_INDEX = 0;
std::size_t DIVISOR_INDEX = 1;

static Tensor divide_tensors(Tensor& dividend, Tensor& divisor) {
    if (xt::any(xt::equal(divisor.data, 0.0))) {
        throw std::runtime_error("Division by zero encountered");
    }

    Tensor quotient = Tensor::from_xarray(xt::eval(dividend.data / divisor.data));  // xtensor handles broadcasting
    auto gradient_fn = new DivBackward(dividend, divisor);
    gradient_fn->saved_tensors.insert(gradient_fn->saved_tensors.begin(), 
                                    {dividend.save(), divisor.save()});
    quotient.gradient_fn = gradient_fn;
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
    edges.push_back(autograd::Edge(0, dividend.get_gradient_edge()));
    edges.push_back(autograd::Edge(1, divisor.get_gradient_edge()));
}

xt::xarray<float> reduce_broadcast(const xt::xarray<float>& grad, 
                                   const xt::xarray<float>& original) {
    auto input_shape = original.shape();
    auto input_rank = input_shape.size();
    auto output_shape = grad.shape();
    auto output_rank = output_shape.size();

    // Align shapes on trailing dimensions and pad input dims if needed
    std::vector<std::size_t> padded_input_shape(output_rank, 1);
    for (std::size_t i = 0; i < input_rank; ++i) {
        padded_input_shape[output_rank - 1 - i] = input_shape[input_rank - 1 - i];
    }

    auto result = grad;
    // Summation from the highest dimension down to 0
    for (int dim = output_rank - 1; dim >= 0; dim--) {
        if (padded_input_shape[dim] != result.shape()[dim]) {
            result = xt::sum(result, dim);
        }
    }
    return result;
}

std::vector<Tensor> DivBackward::operator()(Tensor output_grad) {
    auto dividend = saved_tensors[DIVIDEND_INDEX];
    auto divisor = saved_tensors[DIVISOR_INDEX];

    // For division z = x/y:
    // ∂z/∂x = 1/y
    // ∂z/∂y = -x/y²
    
    // Calculate raw gradients with broadcasting
    auto dividend_grad_raw = xt::eval(output_grad.data / divisor.data);
    auto divisor_grad_raw = xt::eval(output_grad.data * (-dividend.data / (divisor.data * divisor.data)));

    // Reduce gradients along broadcast dimensions
    auto dividend_grad = reduce_broadcast(dividend_grad_raw, dividend.data);
    auto divisor_grad = reduce_broadcast(divisor_grad_raw, divisor.data);

    return {Tensor::from_xarray(dividend_grad), Tensor::from_xarray(divisor_grad)};
}

} // namespace ember
