#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(TensorDivision, ScalarTensorsCanBeDivided) {
    Tensor a = {24.0};
    Tensor b = {6.0};
    Tensor c = a / b;

    auto quotient = c.data;
    auto expected_quotient = xt::xarray<double>({4.0});
    EXPECT_TRUE(xt::allclose(quotient, expected_quotient));

    c.backward();
    // ∂c/∂a = 1/b = 1/6
    // ∂c/∂b = -a/b² = -24/36
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0/6.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{-2.0/3.0}));
}

TEST(TensorDivision, MultidimensionalTensorsCanBeDivided) {
    Tensor a = {{12.0, 9.0}, {6.0, 3.0}};
    Tensor b = {{3.0, 3.0}, {2.0, 1.0}};
    Tensor c = a / b;

    xt::xarray<double> quotient = c.data;
    xt::xarray<double> expected_quotient = {{4.0, 3.0}, {3.0, 3.0}};

    EXPECT_TRUE(xt::allclose(quotient, expected_quotient));

    c.backward();
    // ∂c/∂a = 1/b
    EXPECT_TRUE(xt::allclose(a.gradient->data, 
                            xt::xarray<double>{{1.0/3.0, 1.0/3.0}, 
                                            {1.0/2.0, 1.0/1.0}}));
    // ∂c/∂b = -a/b²
    EXPECT_TRUE(xt::allclose(b.gradient->data, 
                            xt::xarray<double>{{-4.0/3.0, -3.0/3.0}, 
                                            {-3.0/2.0, -3.0/1.0}}));
}

TEST(TensorDivision, AnonymousIntermediateTensorsCanBeDivided) {
    Tensor a = {{8.0, 6.0}, {4.0, 2.0}};
    Tensor b = {{2.0, 2.0}, {2.0, 2.0}};
    Tensor c = a / b;
    
    Tensor d = (c / Tensor({{2.0, 2.0}, {2.0, 2.0}})) / 
               (c / Tensor({{4.0, 4.0}, {4.0, 4.0}}));
    EXPECT_TRUE(xt::allclose(d.data, xt::xarray<double>({{2.0, 2.0}, {2.0, 2.0}})));

    d.backward();
    // Complex chain of divisions should still maintain correct gradients
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>({{0.0, 0.0}, {0.0, 0.0}})));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>({{0.0, 0.0}, {0.0, 0.0}})));
}

TEST(TensorDivision, BroadcastingWorks) {
    Tensor a = {6.0, 9.0, 12.0};
    Tensor b = {3.0};  // Scalar to be broadcast
    Tensor c = a / b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<double>{2.0, 3.0, 4.0}));

    c.backward();
    // ∂c/∂a = 1/b = 1/3 for each element
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0/3.0, 1.0/3.0, 1.0/3.0}));
    // ∂c/∂b = sum(-a/b²) = -(6+9+12)/9 = -3
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{-3.0}));
}

TEST(TensorDivision, DivisionByOne) {
    Tensor a = {1.0, 2.0, 3.0};
    Tensor b = {1.0, 1.0, 1.0};
    Tensor c = a / b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<double>{1.0, 2.0, 3.0}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<double>{1.0, 1.0, 1.0}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<double>{-1.0, -2.0, -3.0}));
}

TEST(TensorDivision, DivisionByZeroThrows) {
    Tensor a = {1.0};
    Tensor b = {0.0};
    EXPECT_THROW(a / b, std::runtime_error);
}

TEST(TensorDivision, GradientWithBroadcastAndScalar) {
    Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor scalar = {2.0};
    
    auto c = a / scalar;
    c.backward();

    // Expected: ∂(a/s)/∂a = 1/s
    xt::xarray<double> expected_grad_a = xt::ones_like(a.data) / scalar.data;
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a, 0.001));

    // Expected: ∂(a/s)/∂s = -a/s²
    xt::xarray<double> expected_grad_scalar = {-xt::sum(a.data / (scalar.data * scalar.data))};
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, expected_grad_scalar, 0.001));
} 
