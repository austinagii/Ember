#ifndef EMBER_OPS_ADD_H
#define EMBER_OPS_ADD_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <xtensor/xarray.hpp>

#include <vector>

namespace ember {

/**
 * @brief Add two tensors and return a new tensor representing the sum.
 */
Tensor add(Tensor& augend, Tensor& addend);

Tensor operator+(Tensor& augend, Tensor& addend);
Tensor operator+(Tensor& augend, Tensor&& addend);
Tensor operator+(Tensor&& augend, Tensor& addend);
Tensor operator+(Tensor&& augend, Tensor&& addend);

}  // namespace ember

#endif  // EMBER_OPS_ADD_H
