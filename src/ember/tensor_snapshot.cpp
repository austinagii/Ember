#include <ember/tensor_snapshot.h>
#include <ember/tensor.h>

namespace ember {

TensorSnapshot::TensorSnapshot(const Tensor* tensor): value(tensor->value) {}

} // namespace ember 