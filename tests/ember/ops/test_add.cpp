#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>        // for operator<<, if you want
#include <xtensor/xfixed.hpp>

#include <iostream>

using namespace ember;

TEST(TensorAddition, ScalarTensorsCanBeAdded) {
  Tensor a = {1.0f};
  Tensor b = {5.0f};
  Tensor c = a + b;

  auto sum = c.data;
  auto expected_sum = xt::xarray<float>({6.0f});
  EXPECT_TRUE(xt::allclose(sum, expected_sum));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{1.0f}));
}

TEST(TensorAddition, MultidimensionalTensorsCanBeAdded) {
  Tensor a = {{1.0f, 9.0f}, {3.0f, 2.2f}};
  Tensor b = {{5.0f, 3.0f}, {2.0f, 1.3f}};
  Tensor c = a + b;

  xt::xarray<float> sum = c.data;
  xt::xarray<float> expected_sum = {{6.0f, 12.0f}, {5.0f, 3.5f}};

  EXPECT_TRUE(xt::allclose(sum, expected_sum));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
}

TEST(TensorAddition, AnonymousIntermediateTensorsCanBeAdded) {
  Tensor a = {{7.0f, 3.0f}, {4.0f, 1.0f}};
  Tensor b = {{8.0f, 2.0f}, {5.0f, 0.0f}};
  Tensor c = a + b;
  
  Tensor d = (c + Tensor({{3.0f, 3.0f}, {3.0f, 3.0f}})) + 
             (c + Tensor({{5.0f, 5.0f}, {5.0f, 5.0f}}));
  EXPECT_TRUE(xt::allclose(d.data, xt::xarray<float>({{38.0f, 18.0f}, {26.0f, 10.0f}})));

  d.backward();
  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>({{2.0f, 2.0f}, {2.0f, 2.0f}})));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>({{2.0f, 2.0f}, {2.0f, 2.0f}})));
}

TEST(TensorAddition, BroadcastingWorks) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {5.0f};  // Scalar to be broadcast
    Tensor c = a + b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{6.0f, 7.0f, 8.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{3.0f}));  // Sum of gradients
}

TEST(TensorAddition, ZeroAdditionPreservesValues) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {0.0f, 0.0f, 0.0f};
    Tensor c = a + b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{1.0f, 2.0f, 3.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
}

TEST(TensorAddition, GradientWithBroadcastAndScalar) {
    Tensor matrix = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor scalar = {2.0f};
    
    auto c = matrix + scalar;
    c.backward();

    // Expected: ∂(m+s)/∂m = 1, ∂(m+s)/∂s = sum(1)
    EXPECT_TRUE(xt::allclose(matrix.gradient->data, xt::ones_like(matrix.data), 0.001f));
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, xt::xarray<float>({4.0f}), 0.001f));
}
