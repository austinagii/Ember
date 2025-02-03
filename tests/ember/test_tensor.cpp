#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <xtensor/xio.hpp>

using namespace ember;

TEST(TensorConstructors, DefaultConstructorCreatesEmptyTensor) {
  Tensor t;
  EXPECT_EQ(t.data_.dimension(), 0);
  EXPECT_EQ(t.data_.size(), 1);
  EXPECT_TRUE(xt::allclose(t.data_, xt::xarray<float>{}));
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}

TEST(TensorConstructors, OneDimensionalInitializerList) {
  Tensor t = {1.0, 2.0, 3.0};
  EXPECT_EQ(t.data_.dimension(), 1);
  EXPECT_EQ(t.data_.shape()[0], 3);
  EXPECT_TRUE(xt::allclose(t.data_, xt::xarray<double>{1.0, 2.0, 3.0}));
}

TEST(TensorConstructors, TwoDimensionalInitializerList) {
  Tensor t = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  EXPECT_EQ(t.data_.dimension(), 2);
  EXPECT_EQ(t.data_.shape()[0], 3);
  EXPECT_EQ(t.data_.shape()[1], 2);
  EXPECT_TRUE(xt::allclose(
      t.data_, xt::xarray<double>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}));
}

TEST(TensorConstructors, ThreeDimensionalInitializerList) {
  Tensor t = {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};
  EXPECT_EQ(t.data_.dimension(), 3);
  EXPECT_EQ(t.data_.shape()[0], 2);
  EXPECT_EQ(t.data_.shape()[1], 2);
  EXPECT_EQ(t.data_.shape()[2], 2);

  xt::xarray<double> expected = {{{1.0, 2.0}, {3.0, 4.0}},
                                 {{5.0, 6.0}, {7.0, 8.0}}};
  EXPECT_TRUE(xt::allclose(t.data_, expected));
}

TEST(TensorStaticInitializers, FromXArrayCreatesCorrectTensor) {
  xt::xarray<double> data = {{1.0, 2.0}, {3.0, 4.0}};
  Tensor t = Tensor::from_xarray_(data);
  EXPECT_TRUE(xt::allclose(t.data_, data));
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}

TEST(TensorStaticInitializers, ZerosLikeCreatesMatchingShape) {
  Tensor original = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  Tensor zeros = Tensor::zeros_like(original);

  EXPECT_EQ(zeros.data_.shape(), original.data_.shape());
  EXPECT_TRUE(xt::allclose(zeros.data_, xt::zeros_like(original.data_)));
  EXPECT_EQ(zeros.gradient, nullptr);
  EXPECT_EQ(zeros.gradient_fn, nullptr);
  EXPECT_EQ(zeros.gradient_accumulator, nullptr);
}

TEST(TensorConstructors, EmptyInitializerListCreatesScalarTensor) {
  Tensor t = {1.0};
  EXPECT_EQ(t.data_.dimension(), 1);
  EXPECT_EQ(t.data_.shape()[0], 1);
  EXPECT_TRUE(xt::allclose(t.data_, xt::xarray<double>{1.0}));
}

TEST(TensorConstructors, InitializerListsPreserveGradientProperties) {
  Tensor t = {{1.0, 2.0}, {3.0, 4.0}};
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.gradient_fn, nullptr);
  EXPECT_EQ(t.gradient_accumulator, nullptr);
}
