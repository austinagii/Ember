#ifndef TENSOR_SNAPSHOT_H
#define TENSOR_SNAPSHOT_H

#include <xtensor/xarray.hpp>

namespace ember {
    struct Tensor;  // Forward declaration

    struct TensorSnapshot {
        xt::xarray<float> data;
        TensorSnapshot(Tensor* tensor);
    };
} // namespace ember

#endif // TENSOR_SNAPSHOT_H 
