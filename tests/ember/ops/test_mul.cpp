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
