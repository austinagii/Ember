#ifndef EMBER_OPS_UTILS_H
#define EMBER_OPS_UTILS_H

#include "xtensor/xarray.hpp"

namespace ember {

xt::xarray<double> reduce_broadcast(
    const xt::xarray<double>& source,
    const xt::xarray<double>::shape_type& target_shape);

}  // namespace ember

#endif  // EMBER_OPS_UTILS_H
