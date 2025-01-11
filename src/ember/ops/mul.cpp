#include <ember/ops/mul.h>

namespace ember {

// Helper function to avoid code duplication
static Tensor multiply_tensors(Tensor& multiplicand, Tensor& multiplier) {
    auto product = Tensor(multiplicand.value * multiplier.value);
    product.gradient_fn = new MulBackward(multiplicand, multiplier);
    return product;
}

// Lvalue & Lvalue
Tensor operator*(Tensor& multiplicand, Tensor& multiplier) {
    return multiply_tensors(multiplicand, multiplier);
}

// Rvalue & Lvalue
Tensor operator*(Tensor&& multiplicand, Tensor& multiplier) {
    return multiply_tensors(multiplicand, multiplier);
}

// Lvalue & Rvalue
Tensor operator*(Tensor& multiplicand, Tensor&& multiplier) {
    return multiply_tensors(multiplicand, multiplier);
}

// Rvalue & Rvalue
Tensor operator*(Tensor&& multiplicand, Tensor&& multiplier) {
    return multiply_tensors(multiplicand, multiplier);
}

MulBackward::MulBackward(Tensor& multiplicand, Tensor& multiplier) {
    saved_tensors.push_back(multiplicand.save());
    saved_tensors.push_back(multiplier.save());

    next_fns.push_back(multiplicand.get_gradient_edge());
    next_fns.push_back(multiplier.get_gradient_edge());
}

std::vector<Tensor> MulBackward::operator()(Tensor output_grad) {
    auto multiplicand_grad = Tensor(saved_tensors[1].value * output_grad.value);
    auto multiplier_grad = Tensor(saved_tensors[0].value * output_grad.value);
    return {multiplicand_grad, multiplier_grad};
}

} // namespace ember
