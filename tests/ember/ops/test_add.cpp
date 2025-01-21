#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>        // for operator<<, if you want
#include <xtensor/xfixed.hpp>

#include <iostream>

using namespace ember;

TEST(TensorAddition, ScalarTensorsCanBeAdded) {
  Tensor a = xt::xarray<float>({1.0f});
  Tensor b = xt::xarray<float>({5.0f});
  Tensor c = a + b;

  auto sum = c.data;
  auto expected_sum = xt::xarray<float>({6.0f});
  EXPECT_TRUE(xt::allclose(sum, expected_sum));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{1.0f}));
}

TEST(TensorAddition, MultidimensionalTensorsCanBeAdded) {
  Tensor a = xt::xarray<float>({{1.0f, 9.0f}, {3.0f, 2.2f}});
  Tensor b = xt::xarray<float>({{5.0f, 3.0f}, {2.0f, 1.3f}});
  Tensor c = a + b;

  xt::xarray<float> sum = c.data;
  xt::xarray<float> expected_sum = {{6.0f, 12.0f}, {5.0f, 3.5f}};

  EXPECT_TRUE(xt::allclose(sum, expected_sum));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
}

TEST(TensorAddition, AnonymousIntermediateTensorsCanBeAdded) {
  Tensor a = xt::xarray<float>({{7.0f, 3.0f}, {4.0f, 1.0f}});
  Tensor b = xt::xarray<float>({{8.0f, 2.0f}, {5.0f, 0.0f}});
  Tensor c = a + b;
  
  Tensor d = (c + Tensor(xt::xarray<float>({{3.0f, 3.0f}, {3.0f, 3.0f}}))) + 
             (c + Tensor(xt::xarray<float>({{5.0f, 5.0f}, {5.0f, 5.0f}})));
  EXPECT_TRUE(xt::allclose(d.data, xt::xarray<float>({{38.0f, 18.0f}, {26.0f, 10.0f}})));

  d.backward();
  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>({{2.0f, 2.0f}, {2.0f, 2.0f}})));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>({{2.0f, 2.0f}, {2.0f, 2.0f}})));
}

TEST(TensorAddition, BroadcastingWorks) {
    Tensor a = xt::xarray<float>({1.0f, 2.0f, 3.0f});
    Tensor b = xt::xarray<float>({5.0f});  // Scalar to be broadcast
    Tensor c = a + b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{6.0f, 7.0f, 8.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{3.0f}));  // Sum of gradients
}

TEST(TensorAddition, ZeroAdditionPreservesValues) {
    Tensor a = xt::xarray<float>({1.0f, 2.0f, 3.0f});
    Tensor b = xt::xarray<float>({0.0f, 0.0f, 0.0f});
    Tensor c = a + b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{1.0f, 2.0f, 3.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
}
