#ifndef EMBER_AUTOGRAD_ACCUMULATOR_H
#define EMBER_AUTOGRAD_ACCUMULATOR_H

#include <ember/tensor.h>
#include <ember/autograd/node.h>

#include <vector>

namespace ember::autograd {

/**
 * @brief Accumulator node for gradient accumulation in the autograd graph.
 * 
 * This node is responsible for accumulating gradients during backpropagation
 * for parameters that require gradients.
 */
class Accumulator final : public Node {
public:
    /**
     * @brief Constructs an Accumulator node.
     * @param target Pointer to the tensor whose gradients need to be accumulated.
     * @throws std::invalid_argument if target is nullptr
     */
    explicit Accumulator(ember::Tensor* target);

    /**
     * @brief Accumulates gradients during backward pass.
     * @param output_grad The incoming gradient from the output
     * @return Vector containing the accumulated gradients
     */
    std::vector<ember::Tensor> operator()(ember::Tensor output_grad) override;

    // Prevent copying and assignment
    Accumulator(const Accumulator&) = delete;
    Accumulator& operator=(const Accumulator&) = delete;

private:
    ember::Tensor* target;  
};

} // namespace ember::autograd

#endif // EMBER_AUTOGRAD_ACCUMULATOR_H
