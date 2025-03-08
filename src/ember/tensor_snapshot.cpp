#include <ember/tensor.h>
#include <ember/tensor_snapshot.h>

namespace ember {

TensorSnapshot::TensorSnapshot(const Tensor& tensor) : data_(tensor.data_) {}

}  // namespace ember