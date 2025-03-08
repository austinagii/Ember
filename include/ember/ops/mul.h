#ifndef EMBER_OPS_MUL_H
#define EMBER_OPS_MUL_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <xtensor/xarray.hpp>

#include <vector>

namespace ember {

/**
 * @brief Multiply two tensors and return a new tensor representing the product.
 */
Tensor mul(Tensor& multiplicand, Tensor& multiplier);

/**
 * @brief Multiply two tensors and return a new tensor representing the product.
 */
Tensor mul(Tensor&& multiplicand, Tensor& multiplier);

/**
 * @brief Multiply two tensors and return a new tensor representing the product.
 */
Tensor mul(Tensor& multiplicand, Tensor&& multiplier);

/**
 * @brief Multiply two tensors and return a new tensor representing the product.
 */
Tensor mul(Tensor&& multiplicand, Tensor&& multiplier);

}  // namespace ember

#endif  // EMBER_OPS_MUL_H
