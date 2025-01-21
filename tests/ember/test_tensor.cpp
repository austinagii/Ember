#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xbroadcast.hpp>

#include <iostream>

using namespace ember;

TEST(Tensor, TensorsCanBeCreatedWithScalarConstants) {
  auto a = Tensor(5.0f);
  EXPECT_EQ(a.value, 5.0f);
}

// TEST(Tensor, TensorCreation) {
//   xt::xarray<float> a = {{1}, {2}};
//   xt::xarray<float> b = {{1, 2, 3}, {1, 2, 3}};
//   auto shape = xt::broadcast(a, b.shape());

//   for (auto it = shape.begin(); it != shape.end(); ++it) {
//     // auto inner = *it;
//       std::cout << *it << " " << std::endl;
//     // for (auto it = inner.begin(); it != inner.end(); ++it) {
//     //   std::cout << *it << " " << std::endl;
//     // }
//     std::cout << std::endl;
//   }
// }
