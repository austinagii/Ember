#ifndef EMBER_OPS_DIV_H
#define EMBER_OPS_DIV_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <xtensor/xarray.hpp>

#include <vector>

namespace ember {

/**
 * @brief Divide two tensors and return a new tensor representing the quotient.
 */
Tensor div(const Tensor& dividend, const Tensor& divisor);

}  // namespace ember

#endif  // EMBER_OPS_DIV_H
