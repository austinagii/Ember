#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xio.hpp>

using namespace ember;

TEST(TensorConstructors, DefaultConstructorCreatesEmptyTensor) {
  Tensor t;
  std::cout << t.data << "\n";
  EXPECT_EQ(t.data.dimension(), 0);
  EXPECT_EQ(t.data.size(), 1);
  EXPECT_TRUE(xt::allclose(t.data, xt::xarray<float>{}));
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}

TEST(TensorConstructors, XArrayConstructorPreservesShape) {
  xt::xarray<float> data = xt::xarray<float>::from_shape({2, 3});
  Tensor t(data);
  EXPECT_EQ(t.data.shape(), data.shape());
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}

TEST(TensorConstructors, OneDimensionalInitializerList) {
  Tensor t = {1.0f, 2.0f, 3.0f};
  EXPECT_EQ(t.data.dimension(), 1);
  EXPECT_EQ(t.data.shape()[0], 3);
  EXPECT_TRUE(xt::allclose(t.data, xt::xarray<float>{1.0f, 2.0f, 3.0f}));
}

TEST(TensorConstructors, TwoDimensionalInitializerList) {
  Tensor t = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  EXPECT_EQ(t.data.dimension(), 2);
  EXPECT_EQ(t.data.shape()[0], 3);
  EXPECT_EQ(t.data.shape()[1], 2);
  EXPECT_TRUE(xt::allclose(t.data, xt::xarray<float>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}));
}

TEST(TensorConstructors, ThreeDimensionalInitializerList) {
  Tensor t = {{{1.0f, 2.0f}, {3.0f, 4.0f}},
              {{5.0f, 6.0f}, {7.0f, 8.0f}}};
  EXPECT_EQ(t.data.dimension(), 3);
  EXPECT_EQ(t.data.shape()[0], 2);
  EXPECT_EQ(t.data.shape()[1], 2);
  EXPECT_EQ(t.data.shape()[2], 2);
  
  xt::xarray<float> expected = {{{1.0f, 2.0f}, {3.0f, 4.0f}},
                               {{5.0f, 6.0f}, {7.0f, 8.0f}}};
  EXPECT_TRUE(xt::allclose(t.data, expected));
}

TEST(TensorStaticInitializers, FromXArrayCreatesCorrectTensor) {
  xt::xarray<float> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  Tensor t = Tensor::from_xarray(data);
  EXPECT_TRUE(xt::allclose(t.data, data));
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}

TEST(TensorStaticInitializers, ZerosLikeCreatesMatchingShape) {
  Tensor original = {{1.0f, 2.0f, 3.0f},
                    {4.0f, 5.0f, 6.0f}};
  Tensor zeros = Tensor::zeros_like(original);
  
  EXPECT_EQ(zeros.data.shape(), original.data.shape());
  EXPECT_TRUE(xt::allclose(zeros.data, xt::zeros_like(original.data)));
  EXPECT_EQ(zeros.gradient, nullptr);
  EXPECT_EQ(zeros.gradient_fn, nullptr);
  EXPECT_EQ(zeros.gradient_accumulator, nullptr);
}

TEST(TensorConstructors, EmptyInitializerListCreatesScalarTensor) {
  Tensor t = {1.0f};
  EXPECT_EQ(t.data.dimension(), 1);
  EXPECT_EQ(t.data.shape()[0], 1);
  EXPECT_TRUE(xt::allclose(t.data, xt::xarray<float>{1.0f}));
}

TEST(TensorConstructors, InitializerListsPreserveGradientProperties) {
  Tensor t = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}
