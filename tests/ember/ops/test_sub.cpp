#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(TensorSubtraction, ScalarTensorsCanBeSubtracted) {
  Tensor a = {1.0};
  Tensor b = {5.0};
  Tensor c = a - b;

  auto difference = c.data;
  auto expected_difference = xt::xarray<double>({-4.0});
  EXPECT_TRUE(xt::allclose(difference, expected_difference));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{-1.0}));
}

TEST(TensorSubtraction, MultidimensionalTensorsCanBeSubtracted) {
  Tensor a = {{1.0, 9.0}, {3.0, 2.2}};
  Tensor b = {{5.0, 3.0}, {2.0, 1.3}};
  Tensor c = a - b;

  xt::xarray<double> difference = c.data;
  xt::xarray<double> expected_difference = {{-4.0, 6.0}, {1.0, 0.9}};

  EXPECT_TRUE(xt::allclose(difference, expected_difference));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{{1.0, 1.0}, {1.0, 1.0}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{{-1.0, -1.0}, {-1.0, -1.0}}));
}

TEST(TensorSubtraction, AnonymousIntermediateTensorsCanBeSubtracted) {
  Tensor a = {{7.0, 3.0}, {4.0, 1.0}};
  Tensor b = {{8.0, 2.0}, {5.0, 0.0}};
  Tensor c = a - b;
  
  Tensor d = (c - Tensor({{3.0, 3.0}, {3.0, 3.0}})) - 
             (c - Tensor({{5.0, 5.0}, {5.0, 5.0}}));
  EXPECT_TRUE(xt::allclose(d.data, xt::xarray<double>({{2.0, 2.0}, {2.0, 2.0}})));

  d.backward();
  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>({{0.0, 0.0}, {0.0, 0.0}})));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>({{0.0, 0.0}, {0.0, 0.0}})));
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
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {1.0, 2.0, 3.0};
    Tensor c = a - b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<double>{0.0, 0.0, 0.0}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0, 1.0, 1.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{-1.0, -1.0, -1.0}));
}

TEST(TensorSubtraction, GradientWithBroadcastAndScalar) {
    Tensor matrix = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor scalar = {2.0};
    
    auto c = matrix - scalar;
    c.backward();

    // Expected: ∂(m-s)/∂m = 1, ∂(m-s)/∂s = -sum(1)
    EXPECT_TRUE(xt::allclose(matrix.gradient->data, xt::ones_like(matrix.data), 0.001));
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, xt::xarray<double>({-4.0}), 0.001));
}

