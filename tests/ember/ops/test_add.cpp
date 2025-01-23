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

TEST(TensorAddition, GradientWithTensorsOfDifferentShapes) {
    Tensor a = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor b = {3.0f};
    
    auto c = a + b;
    c.backward();

    // Expected: ∂(m+s)/∂m = 1, ∂(m+s)/∂s = sum(1)
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::ones_like(a.data), 0.001f));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>({8.0f}), 0.001f));
}

TEST(TensorAddition, ThreeDimensionalWithScalar) {
    Tensor a = {{{1.0f, 2.0f}, {3.0f, 4.0f}},
                {{5.0f, 6.0f}, {7.0f, 8.0f}}};  // 2x2x2 tensor
    Tensor b = {2.0f};  // Scalar
    Tensor c = a + b;

    xt::xarray<float> expected = {{{3.0f, 4.0f}, {5.0f, 6.0f}},
                                 {{7.0f, 8.0f}, {9.0f, 10.0f}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();

    // Gradient for 'a' should be ones
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{{{1.0f, 1.0f}, {1.0f, 1.0f}},
                                                                 {{1.0f, 1.0f}, {1.0f, 1.0f}}}));
    // Gradient for 'b' should be the count of elements in 'a'
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{8.0f}));
}

TEST(TensorAddition, SingleElementBroadcastToLarge) {
    Tensor a({1.0f});  // Simple 1D tensor
    
    auto b_data = xt::xarray<float>::from_shape({2, 2, 2});
    b_data(0,0,0) = 1.0f; b_data(0,0,1) = 2.0f;
    b_data(0,1,0) = 3.0f; b_data(0,1,1) = 4.0f;
    b_data(1,0,0) = 5.0f; b_data(1,0,1) = 6.0f;
    b_data(1,1,0) = 7.0f; b_data(1,1,1) = 8.0f;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a + b;

    xt::xarray<float> expected = {{{2.0f, 3.0f}, {4.0f, 5.0f}},
                                 {{6.0f, 7.0f}, {8.0f, 9.0f}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();
    
    // Gradient for a should be count of elements in b
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{8.0f}));
    
    // Gradient for b should be ones
    xt::xarray<float> ones = {{{1.0f, 1.0f}, {1.0f, 1.0f}},
                             {{1.0f, 1.0f}, {1.0f, 1.0f}}};
    EXPECT_TRUE(xt::allclose(b.gradient->data, ones));
}

// Update ComplexBroadcasting test to match mul version
TEST(TensorAddition, ComplexBroadcasting) {
    auto a_data = xt::xarray<float>::from_shape({2, 2, 1});
    a_data(0,0,0) = 1.0f; a_data(0,1,0) = 2.0f;
    a_data(1,0,0) = 3.0f; a_data(1,1,0) = 4.0f;
    Tensor a = Tensor::from_xarray(a_data);

    auto b_data = xt::xarray<float>::from_shape({1, 1, 3});
    b_data(0,0,0) = 1.0f; b_data(0,0,1) = 2.0f; b_data(0,0,2) = 3.0f;
    Tensor b = Tensor::from_xarray(b_data);

    Tensor c = a + b;  // Should broadcast to 2x2x3

    xt::xarray<float> expected = {{{2.0f, 3.0f, 4.0f},
                                  {3.0f, 4.0f, 5.0f}},
                                 {{4.0f, 5.0f, 6.0f},
                                  {5.0f, 6.0f, 7.0f}}};
    EXPECT_TRUE(xt::allclose(c.data, expected));

    c.backward();
    
    // Gradient for a should sum across broadcast dimension
    auto expected_grad_a = xt::xarray<float>::from_shape({2, 2, 1});
    expected_grad_a.fill(3.0f);  // Sum of ones across broadcast dimension
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a));
    
    // Gradient for b should sum across non-broadcast dimensions
    xt::xarray<float> expected_grad_b = {{4.0f, 4.0f, 4.0f}};  // Sum of ones across non-broadcast dimensions
    EXPECT_TRUE(xt::allclose(b.gradient->data, expected_grad_b));
}
