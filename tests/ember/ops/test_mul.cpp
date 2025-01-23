#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xfixed.hpp>
#include <iostream>

using namespace ember;

TEST(TensorMultiplication, ScalarTensorsCanBeMultiplied) {
  Tensor a = {3.0f};
  Tensor b = {2.0f};
  Tensor c = a * b;

  auto product = c.data;
  auto expected_product = xt::xarray<float>({6.0f});
  EXPECT_TRUE(xt::allclose(product, expected_product));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{2.0f}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{3.0f}));
}

TEST(TensorMultiplication, MultidimensionalTensorsCanBeMultiplied) {
  Tensor a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  Tensor b = {{2.0f, 3.0f}, {4.0f, 5.0f}};
  Tensor c = a * b;

  xt::xarray<float> product = c.data;
  xt::xarray<float> expected_product = {{2.0f, 6.0f}, {12.0f, 20.0f}};

  EXPECT_TRUE(xt::allclose(product, expected_product));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{{2.0f, 3.0f}, {4.0f, 5.0f}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{{1.0f, 2.0f}, {3.0f, 4.0f}}));
}

TEST(TensorMultiplication, AnonymousIntermediateTensorsCanBeMultiplied) {
  Tensor a = {{2.0f, 3.0f}, {4.0f, 5.0f}};
  Tensor b = {{3.0f, 4.0f}, {5.0f, 6.0f}};
  Tensor c = a * b;
  
  Tensor d = (c * Tensor({{2.0f, 2.0f}, {2.0f, 2.0f}})) * 
             (c * Tensor({{3.0f, 3.0f}, {3.0f, 3.0f}}));
  
  xt::xarray<float> expected = {{216.0f, 864.0f}, {2400.0f, 5400.0f}};
  EXPECT_TRUE(xt::allclose(d.data, expected));

  d.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, 
              b.data * (2.0f * c.data * 3.0f + 3.0f * c.data * 2.0f)));
  EXPECT_TRUE(xt::allclose(b.gradient->data, 
              a.data * (2.0f * c.data * 3.0f + 3.0f * c.data * 2.0f)));
}

TEST(TensorMultiplication, BroadcastingWorks) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {2.0f};  // Scalar to be broadcast
    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{2.0f, 4.0f, 6.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{2.0f, 2.0f, 2.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{6.0f}));
}

TEST(TensorMultiplication, MultiplicationByOnePreservesValues) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {1.0f, 1.0f, 1.0f};
    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{1.0f, 2.0f, 3.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{1.0f, 2.0f, 3.0f}));
}

TEST(TensorMultiplication, GradientWithBroadcastAndScalar) {
    Tensor a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor b = {{2.0f, 1.0f}, {0.5f, 2.0f}};
    
    auto c = a * b;
    c.backward();

    // Expected gradients: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    EXPECT_TRUE(xt::allclose(a.gradient->data, b.data, 0.001f));
    EXPECT_TRUE(xt::allclose(b.gradient->data, a.data, 0.001f));
}

TEST(TensorMultiplication, ThreeDimensionalWithScalar) {
    Tensor a = {{{1.0f, 2.0f}, {3.0f, 4.0f}},
                {{5.0f, 6.0f}, {7.0f, 8.0f}}};  // 2x2x2 tensor
    Tensor b = {2.0f};  // Scalar
    Tensor c = a * b;

    xt::xarray<float> expected = {{{2.0f, 4.0f}, {6.0f, 8.0f}},
                                 {{10.0f, 12.0f}, {14.0f, 16.0f}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();

    // Gradient for 'a' should be the scalar value broadcast to match a's shape
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{{{2.0f, 2.0f}, {2.0f, 2.0f}},
                                                                 {{2.0f, 2.0f}, {2.0f, 2.0f}}}));
    // Gradient for 'b' should be the sum of all elements in 'a'
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{36.0f}));
}

TEST(TensorMultiplication, MultiplicationWithZero) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {0.0f};  // Zero scalar
    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{0.0f, 0.0f, 0.0f}));

    c.backward();
    // Gradient for a should be zero
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{0.0f, 0.0f, 0.0f}));
    // Gradient for b should be sum of a
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{6.0f}));
}

TEST(TensorMultiplication, ComplexBroadcasting) {
    auto a_data = xt::xarray<float>::from_shape({2, 2, 1});
    a_data(0,0,0) = 1.0f; a_data(0,1,0) = 2.0f;
    a_data(1,0,0) = 3.0f; a_data(1,1,0) = 4.0f;
    Tensor a = Tensor::from_xarray(a_data);

    auto b_data = xt::xarray<float>::from_shape({1, 1, 3});
    b_data(0,0,0) = 1.0f; b_data(0,0,1) = 2.0f; b_data(0,0,2) = 3.0f;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a * b;  // Should broadcast to 2x2x3

    xt::xarray<float> expected = {{{1.0f, 2.0f, 3.0f},
                                  {2.0f, 4.0f, 6.0f}},
                                 {{3.0f, 6.0f, 9.0f},
                                  {4.0f, 8.0f, 12.0f}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();
    
    // Gradient for a should sum across broadcast dimension
    auto expected_grad_a = xt::xarray<float>::from_shape({2, 2, 1});
    expected_grad_a.fill(6.0f);
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a));
    
    // Gradient for b should sum across non-broadcast dimensions
    xt::xarray<float> expected_grad_b = {{10.0f, 10.0f, 10.0f}};
    EXPECT_TRUE(xt::allclose(b.gradient->data, expected_grad_b));
}

TEST(TensorMultiplication, SingleElementBroadcastToLarge) {
    Tensor a({1.0f});  // Simple 1D tensor is fine with initializer list
    
    auto b_data = xt::xarray<float>::from_shape({2, 2, 2});
    b_data(0,0,0) = 1.0f; b_data(0,0,1) = 2.0f;
    b_data(0,1,0) = 3.0f; b_data(0,1,1) = 4.0f;
    b_data(1,0,0) = 5.0f; b_data(1,0,1) = 6.0f;
    b_data(1,1,0) = 7.0f; b_data(1,1,1) = 8.0f;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a * b;

    EXPECT_TRUE(xt::allclose(c.data, b.data));

    c.backward();
    
    // Gradient for a should be sum of all elements in b
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{36.0f}));
    
    // Gradient for b should be 1.0 broadcast to b's shape
    xt::xarray<float> ones = {{{1.0f, 1.0f}, {1.0f, 1.0f}},
                             {{1.0f, 1.0f}, {1.0f, 1.0f}}};
    EXPECT_TRUE(xt::allclose(b.gradient->data, ones));
}
