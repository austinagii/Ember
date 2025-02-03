#ifndef EMBER_TENSOR_SNAPSHOT_H
#define EMBER_TENSOR_SNAPSHOT_H

#include <xtensor/xarray.hpp>

namespace ember {

struct Tensor;  // Forward declaration

struct TensorSnapshot {
  xt::xarray<double> data_;
  TensorSnapshot(Tensor* tensor);
};
}  // namespace ember

#endif  // EMBER_TENSOR_SNAPSHOT_H
