#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>        // for operator<<, if you want
#include <xtensor/xfixed.hpp>

#include <iostream>

using namespace ember;

TEST(TensorAddition, ScalarTensorsCanBeAdded) {
  Tensor a = {1.0};
  Tensor b = {5.0};
  Tensor c = a + b;

  auto sum = c.data;
  auto expected_sum = xt::xarray<double>({6.0});
  EXPECT_TRUE(xt::allclose(sum, expected_sum));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{1.0}));
}

TEST(TensorAddition, MultidimensionalTensorsCanBeAdded) {
  Tensor a = {{1.0, 9.0}, {3.0, 2.2}};
  Tensor b = {{5.0, 3.0}, {2.0, 1.3}};
  Tensor c = a + b;

  xt::xarray<double> sum = c.data;
  xt::xarray<double> expected_sum = {{6.0, 12.0}, {5.0, 3.5}};

  EXPECT_TRUE(xt::allclose(sum, expected_sum));

  c.backward();

  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{{1.0, 1.0}, {1.0, 1.0}}));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{{1.0, 1.0}, {1.0, 1.0}}));
}

TEST(TensorAddition, AnonymousIntermediateTensorsCanBeAdded) {
  Tensor a = {{7.0f, 3.0f}, {4.0f, 1.0f}};
  Tensor b = {{8.0f, 2.0f}, {5.0f, 0.0f}};
  Tensor c = a + b;
  
  Tensor d = (c + Tensor({{3.0, 3.0}, {3.0, 3.0}})) + 
             (c + Tensor({{5.0, 5.0}, {5.0, 5.0}}));
  EXPECT_TRUE(xt::allclose(d.data, xt::xarray<double>({{38.0, 18.0}, {26.0, 10.0}})));

  d.backward();
  EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>({{2.0, 2.0}, {2.0, 2.0}})));
  EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>({{2.0, 2.0}, {2.0, 2.0}})));
}

TEST(TensorAddition, BroadcastingWorks) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {5.0f};  // Scalar to be broadcast
    Tensor c = a + b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{6.0f, 7.0f, 8.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0, 1.0, 1.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{3.0}));  // Sum of gradients
}

TEST(TensorAddition, ZeroAdditionPreservesValues) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {0.0, 0.0, 0.0};
    Tensor c = a + b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<double>{1.0, 2.0, 3.0}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0, 1.0, 1.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{1.0, 1.0, 1.0}));
}

TEST(TensorAddition, GradientWithBroadcastAndScalar) {
    Tensor matrix = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor scalar = {2.0};
    
    auto c = matrix + scalar;
    c.backward();

    // Expected: ∂(m+s)/∂m = 1, ∂(m+s)/∂s = sum(1)
    EXPECT_TRUE(xt::allclose(matrix.gradient->data, xt::ones_like(matrix.data), 0.001));
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, xt::xarray<double>({4.0}), 0.001));
}

TEST(TensorAddition, GradientWithTensorsOfDifferentShapes) {
    Tensor a = {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};
    Tensor b = {3.0};
    
    auto c = a + b;
    c.backward();

    // Expected: ∂(m+s)/∂m = 1, ∂(m+s)/∂s = sum(1)
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::ones_like(a.data), 0.001));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>({8.0}), 0.001));
}

TEST(TensorAddition, ThreeDimensionalWithScalar) {
    Tensor a = {{{1.0, 2.0}, {3.0, 4.0}},
                {{5.0, 6.0}, {7.0, 8.0}}};  // 2x2x2 tensor
    Tensor b = {2.0};  // Scalar
    Tensor c = a + b;

    xt::xarray<double> expected = {{{3.0, 4.0}, {5.0, 6.0}},
                                 {{7.0, 8.0}, {9.0, 10.0}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();

    // Gradient for 'a' should be ones
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{{{1.0, 1.0}, {1.0, 1.0}},
                                                                 {{1.0, 1.0}, {1.0, 1.0}}}));
    // Gradient for 'b' should be the count of elements in 'a'
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{8.0f}));
}

TEST(TensorAddition, SingleElementBroadcastToLarge) {
    Tensor a({1.0f});  // Simple 1D tensor
    
    auto b_data = xt::xarray<float>::from_shape({2, 2, 2});
    b_data(0,0,0) = 1.0; b_data(0,0,1) = 2.0;
    b_data(0,1,0) = 3.0; b_data(0,1,1) = 4.0;
    b_data(1,0,0) = 5.0; b_data(1,0,1) = 6.0;
    b_data(1,1,0) = 7.0; b_data(1,1,1) = 8.0;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a + b;

    xt::xarray<double> expected = {{{2.0, 3.0}, {4.0, 5.0}},
                                 {{6.0, 7.0}, {8.0, 9.0}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();
    
    // Gradient for a should be count of elements in b
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{8.0}));
    
    // Gradient for b should be ones
    xt::xarray<double> ones = {{{1.0, 1.0}, {1.0, 1.0}},
                             {{1.0, 1.0}, {1.0, 1.0}}};
    EXPECT_TRUE(xt::allclose(b.gradient->data, ones));
}

// Update ComplexBroadcasting test to match mul version
TEST(TensorAddition, ComplexBroadcasting) {
    auto a_data = xt::xarray<float>::from_shape({2, 2, 1});
    a_data(0,0,0) = 1.0; a_data(0,1,0) = 2.0;
    a_data(1,0,0) = 3.0; a_data(1,1,0) = 4.0;
    Tensor a = Tensor::from_xarray(a_data);

    auto b_data = xt::xarray<double>::from_shape({1, 1, 3});
    b_data(0,0,0) = 1.0; b_data(0,0,1) = 2.0; b_data(0,0,2) = 3.0;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a + b;  // Should broadcast to 2x2x3

    xt::xarray<double> expected = {{{2.0, 3.0, 4.0},
                                  {3.0, 4.0, 5.0}},
                                 {{4.0, 5.0, 6.0},
                                  {5.0, 6.0, 7.0}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();
    
    // Gradient for a should sum across broadcast dimension
    auto expected_grad_a = xt::xarray<double>::from_shape({2, 2, 1});
    expected_grad_a.fill(3.0);  // Sum of ones across broadcast dimension
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a));
    
    // Gradient for b should sum across non-broadcast dimensions
    xt::xarray<double> expected_grad_b = {{4.0, 4.0, 4.0}};  // Sum of ones across non-broadcast dimensions
    EXPECT_TRUE(xt::allclose(b.gradient->data, expected_grad_b));
}
