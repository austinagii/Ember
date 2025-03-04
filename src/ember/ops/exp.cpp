#include <ember/ops/exp.h>

namespace ember {

Tensor exp(Tensor exponent) {
  return Tensor::from_xarray_(xt::exp(exponent.data_));
}

}  // namespace ember
