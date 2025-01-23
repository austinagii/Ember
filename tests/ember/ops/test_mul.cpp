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

  auto product = c.data;
  auto expected_product = xt::xarray<double>({6.0});
  EXPECT_TRUE(xt::allclose(product, expected_product));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{2.0}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{3.0}));
}

TEST(TensorMultiplication, MultidimensionalTensorsCanBeMultiplied) {
  Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
  Tensor b = {{2.0, 3.0}, {4.0, 5.0}};
  Tensor c = a * b;

  xt::xarray<double> product = c.data;
  xt::xarray<double> expected_product = {{2.0, 6.0}, {12.0, 20.0}};

  EXPECT_TRUE(xt::allclose(product, expected_product));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{{2.0, 3.0}, {4.0, 5.0}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{{1.0, 2.0}, {3.0, 4.0}}));
}

TEST(TensorMultiplication, AnonymousIntermediateTensorsCanBeMultiplied) {
  Tensor a = {{2.0, 3.0}, {4.0, 5.0}};
  Tensor b = {{3.0, 4.0}, {5.0, 6.0}};
  Tensor c = a * b;
  
  Tensor d = (c * Tensor({{2.0, 2.0}, {2.0, 2.0}})) * 
             (c * Tensor({{3.0, 3.0}, {3.0, 3.0}}));
  
  xt::xarray<double> expected = {{216.0, 864.0}, {2400.0, 5400.0}};
  EXPECT_TRUE(xt::allclose(d.data, expected));

  d.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, 
              b.data * (2.0 * c.data * 3.0 + 3.0 * c.data * 2.0)));
  EXPECT_TRUE(xt::allclose(b.gradient->data, 
              a.data * (2.0 * c.data * 3.0 + 3.0 * c.data * 2.0)));
}

TEST(TensorMultiplication, BroadcastingWorks) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {2.0};  // Scalar to be broadcast
    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<double>{2.0, 4.0, 6.0}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{2.0, 2.0, 2.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{6.0}));
}

TEST(TensorMultiplication, MultiplicationByOnePreservesValues) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {1.0, 1.0, 1.0};
    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<double>{1.0, 2.0, 3.0}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0, 1.0, 1.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{1.0, 2.0, 3.0}));
}

TEST(TensorMultiplication, GradientWithBroadcastAndScalar) {
    Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor b = {{2.0, 1.0}, {0.5, 2.0}};
    
    auto c = a * b;
    c.backward();

    // Expected gradients: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    EXPECT_TRUE(xt::allclose(a.gradient->data, b.data, 0.001));
    EXPECT_TRUE(xt::allclose(b.gradient->data, a.data, 0.001));
}

TEST(TensorMultiplication, ThreeDimensionalWithScalar) {
    Tensor a = {{{1.0, 2.0}, {3.0, 4.0}},
                {{5.0, 6.0}, {7.0, 8.0}}};  // 2x2x2 tensor
    Tensor b = {2.0};  // Scalar
    Tensor c = a * b;

    xt::xarray<double> expected = {{{2.0, 4.0}, {6.0, 8.0}},
                                 {{10.0, 12.0}, {14.0, 16.0}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();

    // Gradient for 'a' should be the scalar value broadcast to match a's shape
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{{{2.0, 2.0}, {2.0, 2.0}},
                                                                 {{2.0, 2.0}, {2.0, 2.0}}}));
    // Gradient for 'b' should be the sum of all elements in 'a'
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{36.0}));
}

TEST(TensorMultiplication, MultiplicationWithZero) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {0.0f};  // Zero scalar
    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{0.0f, 0.0f, 0.0f}));

    c.backward();
    // Gradient for a should be zero
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{0.0, 0.0, 0.0}));
    // Gradient for b should be sum of a
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{6.0f}));
}

TEST(TensorMultiplication, ComplexBroadcasting) {
    auto a_data = xt::xarray<double>::from_shape({2, 2, 1});
    a_data(0,0,0) = 1.0; a_data(0,1,0) = 2.0;
    a_data(1,0,0) = 3.0; a_data(1,1,0) = 4.0;
    Tensor a = Tensor::from_xarray(a_data);

    auto b_data = xt::xarray<double>::from_shape({1, 1, 3});
    b_data(0,0,0) = 1.0; b_data(0,0,1) = 2.0; b_data(0,0,2) = 3.0;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a * b;  // Should broadcast to 2x2x3

    xt::xarray<double> expected = {{{1.0, 2.0, 3.0},
                                  {2.0, 4.0, 6.0}},
                                 {{3.0, 6.0, 9.0},
                                  {4.0, 8.0, 12.0}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();
    
    // Gradient for a should sum across broadcast dimension
    auto expected_grad_a = xt::xarray<double>::from_shape({2, 2, 1});
    expected_grad_a.fill(6.0);
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a));
    
    // Gradient for b should sum across non-broadcast dimensions
    xt::xarray<float> expected_grad_b = {{10.0f, 10.0f, 10.0f}};
    EXPECT_TRUE(xt::allclose(b.gradient->data, expected_grad_b));
}

TEST(TensorMultiplication, SingleElementBroadcastToLarge) {
    Tensor a({1.0f});  // Simple 1D tensor is fine with initializer list
    
    auto b_data = xt::xarray<double>::from_shape({2, 2, 2});
    b_data(0,0,0) = 1.0; b_data(0,0,1) = 2.0;
    b_data(0,1,0) = 3.0; b_data(0,1,1) = 4.0;
    b_data(1,0,0) = 5.0; b_data(1,0,1) = 6.0;
    b_data(1,1,0) = 7.0; b_data(1,1,1) = 8.0;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, b.data));

    c.backward();
    
    // Gradient for a should be sum of all elements in b
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{36.0}));
    
    // Gradient for b should be 1.0 broadcast to b's shape
    xt::xarray<double> ones = {{{1.0, 1.0}, {1.0, 1.0}},
                               {{1.0, 1.0}, {1.0, 1.0}}};
    EXPECT_TRUE(xt::allclose(b.gradient->data, ones));
}
