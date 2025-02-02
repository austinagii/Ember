#include <ember/tensor_snapshot.h>
#include <ember/tensor.h>

namespace ember {

TensorSnapshot::TensorSnapshot(Tensor* tensor): data_(tensor->data_) {}

} // namespace ember 