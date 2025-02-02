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

  EXPECT_EQ(c, Tensor({-4.0}));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor({1.0}));
  EXPECT_EQ(*b.gradient, Tensor({-1.0}));
}

TEST(TensorSubtraction, MultidimensionalTensorsCanBeSubtracted) {
  Tensor a = {{1.0, 9.0}, {3.0, 2.2}};
  Tensor b = {{5.0, 3.0}, {2.0, 1.3}};

  Tensor c = a - b;

  EXPECT_TRUE(c.equals_approx(Tensor({{-4.0, 6.0}, {1.0, 0.9}})));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor({{1.0, 1.0}, {1.0, 1.0}}));
  EXPECT_EQ(*b.gradient, Tensor({{-1.0, -1.0}, {-1.0, -1.0}}));
}

TEST(TensorSubtraction, AnonymousIntermediateTensorsCanBeSubtracted) {
  Tensor a = {{7.0, 3.0}, {4.0, 1.0}};
  Tensor b = {{8.0, 2.0}, {5.0, 0.0}};
  Tensor c = a - b;
  
  Tensor d = (c - Tensor({{3.0, 3.0}, {3.0, 3.0}})) - 
             (c - Tensor({{5.0, 5.0}, {5.0, 5.0}}));
  EXPECT_EQ(d, Tensor({{2.0, 2.0}, {2.0, 2.0}}));

  d.backward();
  EXPECT_EQ(*a.gradient, Tensor({{0.0, 0.0}, {0.0, 0.0}}));
  EXPECT_EQ(*b.gradient, Tensor({{0.0, 0.0}, {0.0, 0.0}}));
}

TEST(TensorSubtraction, BroadcastingWorks) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {5.0};  // Scalar to be broadcast
    Tensor c = a - b;

    EXPECT_EQ(c, Tensor({-4.0, -3.0, -2.0}));

    c.backward();
    EXPECT_EQ(*a.gradient, Tensor({1.0, 1.0, 1.0}));
    EXPECT_EQ(*b.gradient, Tensor({-3.0}));
}

TEST(TensorSubtraction, ZeroSubtractionPreservesGradients) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {1.0, 2.0, 3.0};
    Tensor c = a - b;

    EXPECT_EQ(c, Tensor({0.0, 0.0, 0.0}));

    c.backward();
    EXPECT_EQ(*a.gradient, Tensor({1.0, 1.0, 1.0}));
    EXPECT_EQ(*b.gradient, Tensor({-1.0, -1.0, -1.0}));
}

TEST(TensorSubtraction, GradientWithBroadcastAndScalar) {
    Tensor matrix = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor scalar = {2.0};
    
    auto c = matrix - scalar;
    c.backward();

    // Expected: ∂(m-s)/∂m = 1, ∂(m-s)/∂s = -sum(1)
    EXPECT_EQ(*matrix.gradient, Tensor::ones_like(matrix));
    EXPECT_EQ(*scalar.gradient, Tensor({-4.0}));
}
