#ifndef EMBER_OPS_SUB_H
#define EMBER_OPS_SUB_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <xtensor/xarray.hpp>

#include <vector>

namespace ember {

/**
 * @brief Subtract two tensors and return a new tensor representing the
 * difference.
 */
Tensor sub(Tensor& minuend, Tensor& subtrahend);

/**
 * @brief Subtract two tensors and return a new tensor representing the
 * difference.
 */
Tensor sub(Tensor&& minuend, Tensor& subtrahend);

/**
 * @brief Subtract two tensors and return a new tensor representing the
 * difference.
 */
Tensor sub(Tensor& minuend, Tensor&& subtrahend);

/**
 * @brief Subtract two tensors and return a new tensor representing the
 * difference.
 */
Tensor sub(Tensor&& minuend, Tensor&& subtrahend);

}  // namespace ember

#endif  // EMBER_OPS_SUB_H
