#ifndef EMBER_OPS_MATMUL_H
#define EMBER_OPS_MATMUL_H

#include <ember/autograd/node.h>
#include <ember/tensor.h>

namespace ember {

/**
 * Performs matrix multiplication of a and b.
 */
Tensor matmul(Tensor& a, Tensor& b);

}  // namespace ember

#endif  // EMBER_OPS_MATMUL_H
