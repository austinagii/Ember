#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>        // for operator<<, if you want
#include <xtensor/xfixed.hpp>

#include <iostream>

using namespace ember;

TEST(TensorAddition, ScalarTensorsCanBeAddedTogether) {
  Tensor a = xt::xarray<float>({1.0f});
  Tensor b = xt::xarray<float>({5.0f});
  Tensor c = a + b;

  auto sum = c.data;
  auto expected_sum = xt::xarray<float>({6.0f});

  EXPECT_TRUE(xt::all(xt::equal(sum, expected_sum)));
}

TEST(TensorAddition, MultidimensionalTensorsCanBeAddedTogether) {
  Tensor a = xt::xarray<float>({{1.0f, 9.0f}, {3.0f, 2.2f}});
  Tensor b = xt::xarray<float>({{5.0f, 3.0f}, {2.0f, 1.3f}});
  Tensor c = a + b;

  xt::xarray<float> sum = c.data;
  xt::xarray<float> expected_sum = {{6.0f, 12.0f}, {5.0f, 3.5f}};

  EXPECT_TRUE(xt::all(xt::equal(sum, expected_sum)));
}

// TODO: Reintroduce this test case when the `Tensor` scalar constructor can be 
// updated to create an xt::array from that constant without breaking the other 
// tensor operations that do not support operations on multidimensional arrays.
// TEST(TensorAddition, TensorsCanBeAddedToConstants) {
//   Tensor a(3.0f);
//   auto b = a + 5;
//   EXPECT_EQ(b.value, 8.0f);
// }

TEST(TensorAddition, AnonymousIntermediateTensorsCanBeAddedTogether) {
  Tensor a = xt::xarray<float>({7.0f});
  Tensor b = xt::xarray<float>({8.0f});
  Tensor c = a + b;
  
  Tensor d = (c + Tensor(xt::xarray<float>({3}))) + (c + Tensor(xt::xarray<float>({5})));
  EXPECT_TRUE(xt::all(xt::equal(d.data, xt::xarray<float>({38.0f}))));

  d.backward();
  EXPECT_TRUE(xt::all(xt::equal(a.gradient->data, xt::xarray<float>({2.0f}))));
  EXPECT_TRUE(xt::all(xt::equal(b.gradient->data, xt::xarray<float>({2.0f}))));
}
