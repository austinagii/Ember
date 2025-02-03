#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>  // for operator<<, if you want

#include <iostream>

using namespace ember;

TEST(TensorAddition, ScalarTensorsCanBeAdded) {
  Tensor a({1.0}, true);
  Tensor b({5.0}, true);

  Tensor c = a + b;

  EXPECT_EQ(c, Tensor(6.0));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor(1.0));
  EXPECT_EQ(*b.gradient, Tensor(1.0));
}

TEST(TensorAddition, MultidimensionalTensorsCanBeAdded) {
  Tensor a({{1.0, 9.0}, {3.0, 2.2}}, true);
  Tensor b({{5.0, 3.0}, {2.0, 1.3}}, true);

  Tensor c = a + b;

  EXPECT_EQ(c, Tensor({{6.0, 12.0}, {5.0, 3.5}}));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor({{1.0, 1.0}, {1.0, 1.0}}));
  EXPECT_EQ(*b.gradient, Tensor({{1.0, 1.0}, {1.0, 1.0}}));
}

TEST(TensorAddition, AnonymousIntermediateTensorsCanBeAdded) {
  Tensor a({{7.0f, 3.0f}, {4.0f, 1.0f}}, true);
  Tensor b({{8.0f, 2.0f}, {5.0f, 0.0f}}, true);

  Tensor c = a + b;
  Tensor d = (c + Tensor({{3.0, 3.0}, {3.0, 3.0}})) +
             (c + Tensor({{5.0, 5.0}, {5.0, 5.0}}));

  EXPECT_EQ(d, Tensor({{38.0, 18.0}, {26.0, 10.0}}));

  d.backward();

  EXPECT_EQ(*a.gradient, Tensor({{2.0, 2.0}, {2.0, 2.0}}));
  EXPECT_EQ(*b.gradient, Tensor({{2.0, 2.0}, {2.0, 2.0}}));
}

TEST(TensorAddition, BroadcastingWorks) {
  Tensor a({1.0f, 2.0f, 3.0f}, true);
  Tensor b({5.0f}, true);

  Tensor c = a + b;

  EXPECT_EQ(c, Tensor({6.0f, 7.0f, 8.0f}));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor({1.0, 1.0, 1.0}));
  EXPECT_EQ(*b.gradient, Tensor({3.0}));
}

TEST(TensorAddition, ZeroAdditionPreservesValues) {
  Tensor a({1.0, 2.0, 3.0}, true);
  Tensor b({0.0, 0.0, 0.0}, true);

  Tensor c = a + b;

  EXPECT_EQ(c, Tensor({1.0, 2.0, 3.0}));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor({1.0, 1.0, 1.0}));
  EXPECT_EQ(*b.gradient, Tensor({1.0, 1.0, 1.0}));
}

TEST(TensorAddition, GradientWithBroadcastAndScalar) {
  Tensor matrix({{1.0, 2.0}, {3.0, 4.0}}, true);
  Tensor scalar({2.0}, true);

  auto c = matrix + scalar;
  c.backward();

  EXPECT_EQ(*matrix.gradient, Tensor({{1.0, 1.0}, {1.0, 1.0}}));
  EXPECT_EQ(*scalar.gradient, Tensor({4.0}));
}

TEST(TensorAddition, GradientWithTensorsOfDifferentShapes) {
  Tensor a({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}, true);
  Tensor b({3.0}, true);

  auto c = a + b;
  c.backward();

  EXPECT_EQ(*a.gradient, Tensor::ones_like(a));
  EXPECT_EQ(*b.gradient, Tensor({8.0}));
}

TEST(TensorAddition, ThreeDimensionalWithScalar) {
  Tensor a({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}, true);
  Tensor b({2.0}, true);

  Tensor c = a + b;

  EXPECT_EQ(c, Tensor({{{3.0, 4.0}, {5.0, 6.0}}, {{7.0, 8.0}, {9.0, 10.0}}}));

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor::ones_like(a));
  EXPECT_EQ(*b.gradient, Tensor({8.0}));
}

TEST(TensorAddition, SingleElementBroadcastToLarge) {
  Tensor a({1.0f}, true);

  Tensor b = Tensor::from_shape({2, 2, 2});
  b(0, 0, 0) = 1.0;
  b(0, 0, 1) = 2.0;
  b(0, 1, 0) = 3.0;
  b(0, 1, 1) = 4.0;
  b(1, 0, 0) = 5.0;
  b(1, 0, 1) = 6.0;
  b(1, 1, 0) = 7.0;
  b(1, 1, 1) = 8.0;
  b.requires_grad = true;

  Tensor c = a + b;
  Tensor expected = {{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}};

  EXPECT_EQ(c, expected);

  c.backward();

  EXPECT_EQ(*a.gradient, Tensor({8.0}));
  EXPECT_EQ(*b.gradient, Tensor::ones_like(b));
}

// Update ComplexBroadcasting test to match mul version
TEST(TensorAddition, ComplexBroadcasting) {
  Tensor a = Tensor::from_shape({2, 2, 1});
  a(0, 0, 0) = 1.0;
  a(0, 1, 0) = 2.0;
  a(1, 0, 0) = 3.0;
  a(1, 1, 0) = 4.0;
  a.requires_grad = true;

  Tensor b = Tensor::from_shape({1, 1, 3});
  b(0, 0, 0) = 1.0;
  b(0, 0, 1) = 2.0;
  b(0, 0, 2) = 3.0;
  b.requires_grad = true;

  Tensor c = a + b;

  Tensor expected = {{{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}},
                     {{4.0, 5.0, 6.0}, {5.0, 6.0, 7.0}}};
  EXPECT_EQ(c, expected);

  c.backward();

  Tensor expected_grad_a = Tensor::from_shape({2, 2, 1});
  expected_grad_a(0, 0, 0) = 3.0;
  expected_grad_a(0, 1, 0) = 3.0;
  expected_grad_a(1, 0, 0) = 3.0;
  expected_grad_a(1, 1, 0) = 3.0;
  EXPECT_EQ(*a.gradient, expected_grad_a);

  Tensor expected_grad_b = Tensor::from_shape({1, 1, 3});
  expected_grad_b(0, 0, 0) = 4.0;
  expected_grad_b(0, 0, 1) = 4.0;
  expected_grad_b(0, 0, 2) = 4.0;
  EXPECT_EQ(*b.gradient, expected_grad_b);
}
