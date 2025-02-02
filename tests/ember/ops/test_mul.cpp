#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xfixed.hpp>
#include <iostream>

using namespace ember;

TEST(TensorMultiplication, ScalarTensorsCanBeMultiplied) {
  Tensor a = {3.0};
  Tensor b = {2.0};

  Tensor c = a * b;

  EXPECT_TRUE(c.equals_approx(Tensor({6.0})));

  c.backward();

  EXPECT_TRUE(a.gradient->equals_approx(Tensor{2.0}));
  EXPECT_TRUE(b.gradient->equals_approx(Tensor{3.0}));
}

TEST(TensorMultiplication, MultidimensionalTensorsCanBeMultiplied) {
  Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
  Tensor b = {{2.0, 3.0}, {4.0, 5.0}};
  Tensor c = a * b;

  EXPECT_TRUE(c.equals_approx(Tensor{{2.0, 6.0}, {12.0, 20.0}}));

  c.backward();

  EXPECT_TRUE(a.gradient->equals_approx(Tensor{{2.0, 3.0}, {4.0, 5.0}}));
  EXPECT_TRUE(b.gradient->equals_approx(Tensor{{1.0, 2.0}, {3.0, 4.0}}));
}

TEST(TensorMultiplication, AnonymousIntermediateTensorsCanBeMultiplied) {
  Tensor a = {{2.0, 3.0}, {4.0, 5.0}};
  Tensor b = {{3.0, 4.0}, {5.0, 6.0}};
  Tensor c = a * b;
  
  Tensor d = (c * Tensor({{2.0, 2.0}, {2.0, 2.0}})) * 
             (c * Tensor({{3.0, 3.0}, {3.0, 3.0}}));
  
  EXPECT_TRUE(d.equals_approx(Tensor{{216.0, 864.0}, {2400.0, 5400.0}}));

  d.backward();

  EXPECT_TRUE(a.gradient->equals_approx(
              b * (2.0 * c * 3.0 + 3.0 * c * 2.0)));
  EXPECT_TRUE(b.gradient->equals_approx(
              a * (2.0 * c * 3.0 + 3.0 * c * 2.0)));
}

TEST(TensorMultiplication, BroadcastingWorks) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {2.0};  // Scalar to be broadcast
    Tensor c = a * b;

    EXPECT_TRUE(c.equals_approx(Tensor{2.0, 4.0, 6.0}));

    c.backward();
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{2.0, 2.0, 2.0}));
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{6.0}));
}

TEST(TensorMultiplication, MultiplicationByOnePreservesValues) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {1.0, 1.0, 1.0};
    Tensor c = a * b;

    EXPECT_TRUE(c.equals_approx(Tensor{1.0, 2.0, 3.0}));

    c.backward();
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{1.0, 1.0, 1.0}));
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{1.0, 2.0, 3.0}));
}

TEST(TensorMultiplication, GradientWithBroadcastAndScalar) {
    Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor b = {{2.0, 1.0}, {0.5, 2.0}};
    
    auto c = a * b;
    c.backward();

    // Expected gradients: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    EXPECT_TRUE(a.gradient->equals_approx(b));
    EXPECT_TRUE(b.gradient->equals_approx(a));
}

TEST(TensorMultiplication, ThreeDimensionalWithScalar) {
    Tensor a = {{{1.0, 2.0}, {3.0, 4.0}},
                {{5.0, 6.0}, {7.0, 8.0}}};  // 2x2x2 tensor
    Tensor b = {2.0};  // Scalar
    Tensor c = a * b;

    EXPECT_TRUE(c.equals_approx(Tensor{{{2.0, 4.0}, {6.0, 8.0}},
                                 {{10.0, 12.0}, {14.0, 16.0}}}));

    c.backward();

    // Gradient for 'a' should be the scalar value broadcast to match a's shape
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{{{2.0, 2.0}, {2.0, 2.0}},
                                                                 {{2.0, 2.0}, {2.0, 2.0}}}));
    // Gradient for 'b' should be the sum of all elements in 'a'
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{36.0}));
}

TEST(TensorMultiplication, MultiplicationWithZero) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {0.0f};  // Zero scalar
    Tensor c = a * b;

    EXPECT_TRUE(c.equals_approx(Tensor{0.0f, 0.0f, 0.0f}));

    c.backward();
    // Gradient for a should be zero
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{0.0, 0.0, 0.0}));
    // Gradient for b should be sum of a
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{6.0f}));
}

TEST(TensorMultiplication, ComplexBroadcasting) {
    Tensor a = Tensor::from_shape({2, 2, 1});
    a(0,0,0) = 1.0; a(0,1,0) = 2.0;
    a(1,0,0) = 3.0; a(1,1,0) = 4.0;

    Tensor b = Tensor::from_shape({1, 1, 3});
    b(0,0,0) = 1.0; b(0,0,1) = 2.0; b(0,0,2) = 3.0;

    Tensor c = a * b;

    EXPECT_TRUE(c.equals_approx(Tensor{{{1.0, 2.0, 3.0},
                                        {2.0, 4.0, 6.0}},
                                       {{3.0, 6.0, 9.0},
                                        {4.0, 8.0, 12.0}}}));

    c.backward();
    
    // Gradient for a should sum across broadcast dimension
    Tensor expected_grad_a = Tensor::from_shape({2, 2, 1});
    expected_grad_a(0,0,0) = 6.0; expected_grad_a(0,1,0) = 6.0;
    expected_grad_a(1,0,0) = 6.0; expected_grad_a(1,1,0) = 6.0;
    EXPECT_TRUE(a.gradient->equals_approx(expected_grad_a));

    // Gradient for b should sum across non-broadcast dimensions
    Tensor expected_grad_b = Tensor::from_shape({1, 1, 3});
    expected_grad_b(0,0,0) = 10.0; expected_grad_b(0,0,1) = 10.0; expected_grad_b(0,0,2) = 10.0;
    EXPECT_TRUE(b.gradient->equals_approx(expected_grad_b));
}

TEST(TensorMultiplication, SingleElementBroadcastToLarge) {
    Tensor a({1.0f});  // Simple 1D tensor is fine with initializer list
    
    Tensor b = Tensor::from_shape({2, 2, 2});
    b(0,0,0) = 1.0; b(0,0,1) = 2.0;
    b(0,1,0) = 3.0; b(0,1,1) = 4.0;
    b(1,0,0) = 5.0; b(1,0,1) = 6.0;
    b(1,1,0) = 7.0; b(1,1,1) = 8.0;

    Tensor c = a * b;

    EXPECT_TRUE(c.equals_approx(b));

    c.backward();
    
    // Gradient for a should be sum of all elements in b
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{36.0}));
    
    // Gradient for b should be 1.0 broadcast to b's shape
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{{{1.0, 1.0}, {1.0, 1.0}},
                                                 {{1.0, 1.0}, {1.0, 1.0}}}));
}
