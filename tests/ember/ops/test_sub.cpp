#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(TensorSubtraction, ScalarTensorsCanBeSubtracted) {
  Tensor a = {1.0f};
  Tensor b = {5.0f};
  Tensor c = a - b;

  auto difference = c.data;
  auto expected_difference = xt::xarray<float>({-4.0f});
  EXPECT_TRUE(xt::allclose(difference, expected_difference));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{-1.0f}));
}

TEST(TensorSubtraction, MultidimensionalTensorsCanBeSubtracted) {
  Tensor a = {{1.0f, 9.0f}, {3.0f, 2.2f}};
  Tensor b = {{5.0f, 3.0f}, {2.0f, 1.3f}};
  Tensor c = a - b;

  xt::xarray<float> difference = c.data;
  xt::xarray<float> expected_difference = {{-4.0f, 6.0f}, {1.0f, 0.9f}};

  EXPECT_TRUE(xt::allclose(difference, expected_difference));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{{1.0f, 1.0f}, {1.0f, 1.0f}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{{-1.0f, -1.0f}, {-1.0f, -1.0f}}));
}

TEST(TensorSubtraction, AnonymousIntermediateTensorsCanBeSubtracted) {
  Tensor a = {{7.0f, 3.0f}, {4.0f, 1.0f}};
  Tensor b = {{8.0f, 2.0f}, {5.0f, 0.0f}};
  Tensor c = a - b;
  
  Tensor d = (c - Tensor({{3.0f, 3.0f}, {3.0f, 3.0f}})) - 
             (c - Tensor({{5.0f, 5.0f}, {5.0f, 5.0f}}));
  EXPECT_TRUE(xt::allclose(d.data, xt::xarray<float>({{2.0f, 2.0f}, {2.0f, 2.0f}})));

  d.backward();
  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>({{0.0f, 0.0f}, {0.0f, 0.0f}})));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>({{0.0f, 0.0f}, {0.0f, 0.0f}})));
}

TEST(TensorSubtraction, BroadcastingWorks) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {5.0f};  // Scalar to be broadcast
    Tensor c = a - b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{-4.0f, -3.0f, -2.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{-3.0f}));
}

TEST(TensorSubtraction, ZeroSubtractionPreservesGradients) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {1.0f, 2.0f, 3.0f};
    Tensor c = a - b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{0.0f, 0.0f, 0.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{-1.0f, -1.0f, -1.0f}));
}

TEST(TensorSubtraction, GradientWithBroadcastAndScalar) {
    Tensor matrix = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor scalar = {2.0f};
    
    auto c = matrix - scalar;
    c.backward();

    // Expected: ∂(m-s)/∂m = 1, ∂(m-s)/∂s = -sum(1)
    EXPECT_TRUE(xt::allclose(matrix.gradient->data, xt::ones_like(matrix.data), 0.001f));
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, xt::xarray<float>({-4.0f}), 0.001f));
}

