#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(TensorDivision, ScalarTensorsCanBeDivided) {
    Tensor a = {24.0f};
    Tensor b = {6.0f};
    Tensor c = a / b;

    auto quotient = c.data;
    auto expected_quotient = xt::xarray<float>({4.0f});
    EXPECT_TRUE(xt::allclose(quotient, expected_quotient));

    c.backward();
    // ∂c/∂a = 1/b = 1/6
    // ∂c/∂b = -a/b² = -24/36
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f/6.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{-2.0f/3.0f}));
}

TEST(TensorDivision, MultidimensionalTensorsCanBeDivided) {
    Tensor a = {{12.0f, 9.0f}, {6.0f, 3.0f}};
    Tensor b = {{3.0f, 3.0f}, {2.0f, 1.0f}};
    Tensor c = a / b;

    xt::xarray<float> quotient = c.data;
    xt::xarray<float> expected_quotient = {{4.0f, 3.0f}, {3.0f, 3.0f}};

    EXPECT_TRUE(xt::allclose(quotient, expected_quotient));

    c.backward();
    // ∂c/∂a = 1/b
    EXPECT_TRUE(xt::allclose(a.gradient->data, 
                            xt::xarray<float>{{1.0f/3.0f, 1.0f/3.0f}, 
                                            {1.0f/2.0f, 1.0f/1.0f}}));
    // ∂c/∂b = -a/b²
    EXPECT_TRUE(xt::allclose(b.gradient->data, 
                            xt::xarray<float>{{-4.0f/3.0f, -3.0f/3.0f}, 
                                            {-3.0f/2.0f, -3.0f/1.0f}}));
}

TEST(TensorDivision, AnonymousIntermediateTensorsCanBeDivided) {
    Tensor a = {{8.0f, 6.0f}, {4.0f, 2.0f}};
    Tensor b = {{2.0f, 2.0f}, {2.0f, 2.0f}};
    Tensor c = a / b;
    
    Tensor d = (c / Tensor({{2.0f, 2.0f}, {2.0f, 2.0f}})) / 
               (c / Tensor({{4.0f, 4.0f}, {4.0f, 4.0f}}));
    EXPECT_TRUE(xt::allclose(d.data, xt::xarray<float>({{2.0f, 2.0f}, {2.0f, 2.0f}})));

    d.backward();
    // Complex chain of divisions should still maintain correct gradients
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>({{0.0f, 0.0f}, {0.0f, 0.0f}})));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>({{0.0f, 0.0f}, {0.0f, 0.0f}})));
}

TEST(TensorDivision, BroadcastingWorks) {
    Tensor a = {6.0f, 9.0f, 12.0f};
    Tensor b = {3.0f};  // Scalar to be broadcast
    Tensor c = a / b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{2.0f, 3.0f, 4.0f}));

    c.backward();
    // ∂c/∂a = 1/b = 1/3 for each element
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f}));
    // ∂c/∂b = sum(-a/b²) = -(6+9+12)/9 = -3
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{-3.0f}));
}

TEST(TensorDivision, DivisionByOne) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {1.0f, 1.0f, 1.0f};
    Tensor c = a / b;

    EXPECT_TRUE(xt::allclose(c.data, xt::xarray<float>{1.0f, 2.0f, 3.0f}));

    c.backward();
    EXPECT_TRUE(xt::allclose(a.gradient->data, xt::xarray<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(xt::allclose(b.gradient->data, xt::xarray<float>{-1.0f, -2.0f, -3.0f}));
}

TEST(TensorDivision, DivisionByZeroThrows) {
    Tensor a = {1.0f};
    Tensor b = {0.0f};
    EXPECT_THROW(a / b, std::runtime_error);
} 
