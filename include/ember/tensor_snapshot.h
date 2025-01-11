#ifndef TENSOR_SNAPSHOT_H
#define TENSOR_SNAPSHOT_H

namespace ember {
    struct Tensor;  // Forward declaration

    struct TensorSnapshot {
        float value;
        TensorSnapshot(Tensor* tensor);
    };
} // namespace ember

#endif // TENSOR_SNAPSHOT_H 
