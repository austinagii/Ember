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
  EXPECT_EQ(t.get_gradient_fn(), nullptr);
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
  EXPECT_EQ(t.get_gradient_fn(), nullptr);
}

TEST(TensorStaticInitializers, ZerosLikeCreatesMatchingShape) {
  Tensor original = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  Tensor zeros = Tensor::zeros_like(original);

  EXPECT_EQ(zeros.data_.shape(), original.data_.shape());
  EXPECT_TRUE(xt::allclose(zeros.data_, xt::zeros_like(original.data_)));
  EXPECT_EQ(zeros.gradient, nullptr);
  EXPECT_EQ(zeros.get_gradient_fn(), nullptr);
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
  EXPECT_EQ(t.get_gradient_fn(), nullptr);
}

TEST(TensorCreation, SuccessfullyCreatesTensorWithRandomValues) {
  Tensor t = Tensor::randn({5, 10});
  EXPECT_EQ(t.data_.dimension(), 2);
  EXPECT_EQ(t.data_.shape()[0], 5);
  EXPECT_EQ(t.data_.shape()[1], 10);

  // Check that the values are random
  EXPECT_NE(t.data_(0, 0), t.data_(0, 1));
  // Check that all values are non-zero using elementwise comparison
  EXPECT_TRUE(xt::all(xt::not_equal(t.data_, 0.0)));
}

TEST(TensorCreation, RandnProducesExpectedDistribution) {
  Tensor t = Tensor::randn({1000}, 5.0, 2.0);

  // Calculate mean and check it's close to expected
  double mean = xt::mean(t.data_)();
  EXPECT_NEAR(mean, 5.0, 0.2);  // Allow some deviation due to randomness

  // Calculate std and check it's close to expected
  double variance = xt::mean(xt::pow(t.data_ - mean, 2))();
  double std_dev = std::sqrt(variance);
  EXPECT_NEAR(std_dev, 2.0, 0.2);
}

TEST(TensorCreation, RandnHandlesEmptyShape) {
  Tensor t = Tensor::randn({});
  EXPECT_EQ(t.data_.dimension(), 0);
  EXPECT_EQ(t.data_.size(), 1);
}

TEST(TensorCreation, RandnPreservesGradientProperties) {
  Tensor t = Tensor::randn({2, 2}, 0.0, 1.0);
  EXPECT_EQ(t.gradient, nullptr);
  EXPECT_EQ(t.get_gradient_fn(), nullptr);
  EXPECT_FALSE(t.requires_grad());
}

TEST(TensorCreation, RandnShapeIsCorrect) {
  Tensor t = Tensor::randn({3, 4, 5});
  EXPECT_EQ(t.data_.dimension(), 3);
  EXPECT_EQ(t.data_.shape()[0], 3);
  EXPECT_EQ(t.data_.shape()[1], 4);
  EXPECT_EQ(t.data_.shape()[2], 5);
}