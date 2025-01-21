#include <ember/tensor_snapshot.h>
#include <ember/tensor.h>

namespace ember {

TensorSnapshot::TensorSnapshot(Tensor* tensor): data(tensor->data) {}

} // namespace ember 
