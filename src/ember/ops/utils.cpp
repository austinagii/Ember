#include "ember/ops/utils.h"

namespace ember {

/**
 * Reduces an xarray to a specified target shape by summing over dimensions
 * that were added during broadcasting.
 *
 * This function is used to revert an xarray from its broadcasted shape
 * back to its original shape by summing over the extra dimensions.
 *
 * @param broadcasted_array The xarray that has been broadcasted.
 * @param desired_shape The desired shape to reduce the xarray to.
 * @return A new xarray reduced to the target shape.
 * @throws std::invalid_argument if the target shape has more dimensions
 *         than the source shape.
 */
xt::xarray<double> reduce_broadcast(
    const xt::xarray<double>& broadcasted_array,
    const xt::xarray<double>::shape_type& desired_shape) {
  auto source_shape = broadcasted_array.shape();
  if (desired_shape.size() > source_shape.size()) {
    throw std::invalid_argument(
        "Target shape has more dimensions than source shape.");
  }

  // Create an aligned shape by aligning the desired shape with the
  // trailing dimensions of the source shape. If the desired shape has fewer
  // dimensions, pad with 1's on the left. For example:
  //   source shape:        [4, 1, 3, 3]
  //   desired shape:             [1, 3] // aligns with trailing dimensions
  //   aligned shape:       [1, 1, 1, 3] // padded with ones on the left
  std::vector<std::size_t> aligned_shape(source_shape.size(), 1);
  std::size_t target_offset = source_shape.size() - desired_shape.size();
  for (std::size_t i = 0; i < desired_shape.size(); ++i) {
    aligned_shape[target_offset + i] = desired_shape[i];
  }

  // Make a deep copy of the broadcasted array to perform the reduction.
  xt::xarray<double> result = xt::eval(broadcasted_array);

  // Determine which axes need to be summed over. These are the axes where
  // the source shape differs from the aligned shape.
  std::vector<std::size_t> reduction_axes;
  for (std::size_t i = 0; i < source_shape.size(); ++i) {
    if (source_shape[i] != aligned_shape[i]) {
      reduction_axes.push_back(i);
    }
  }

  // Sum over the identified axes to reduce the array to the desired shape.
  if (!reduction_axes.empty()) {
    result = xt::sum(result, reduction_axes);
  }

  return result;
}

}  // namespace ember
