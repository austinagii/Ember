#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(TensorDivision, ScalarTensorsCanBeDivided) {
    Tensor a({24.0}, true);
    Tensor b({6.0}, true);

    Tensor c = a / b;

    Tensor expected_quotient = {4.0};
    EXPECT_TRUE(c.equals_approx(expected_quotient));

    c.backward();
    // ∂c/∂a = 1/b = 1/6
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{1.0/6.0}));
    // ∂c/∂b = -a/b² = -24/36
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{-24.0/36.0}));
}

TEST(TensorDivision, MultidimensionalTensorsCanBeDivided) {
    Tensor a({{12.0, 9.0}, {6.0, 3.0}}, true);
    Tensor b({{3.0, 3.0}, {2.0, 1.0}}, true);
    Tensor c = a / b;

    Tensor expected_quotient = {{4.0, 3.0}, {3.0, 3.0}};
    EXPECT_TRUE(c.equals_approx(expected_quotient));

    c.backward();
    // ∂c/∂a = 1/b
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{{1.0/3.0, 1.0/3.0}, 
                                                 {1.0/2.0, 1.0/1.0}}));
    // ∂c/∂b = -a/b²
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{{-4.0/3.0, -3.0/3.0}, 
                                                 {-3.0/2.0, -3.0/1.0}}));
}

TEST(TensorDivision, AnonymousIntermediateTensorsCanBeDivided) {
    Tensor a({{8.0, 6.0}, {4.0, 2.0}}, true);
    Tensor b({{2.0, 2.0}, {2.0, 2.0}}, true);

    Tensor c = a / b;
    
    Tensor d = (c / Tensor({{2.0, 2.0}, {2.0, 2.0}})) / 
               (c / Tensor({{4.0, 4.0}, {4.0, 4.0}}));
    EXPECT_TRUE(d.equals_approx(Tensor{{2.0, 2.0}, {2.0, 2.0}}));

    d.backward();
    // ∂d/∂a = 0
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{{0.0, 0.0}, {0.0, 0.0}}));
    // ∂d/∂b = 0
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{{0.0, 0.0}, {0.0, 0.0}}));
}

TEST(TensorDivision, BroadcastingWorks) {
    Tensor a({6.0, 9.0, 12.0}, true);
    Tensor b({3.0}, true);  // Scalar to be broadcast
    Tensor c = a / b;

    EXPECT_TRUE(c.equals_approx(Tensor{2.0, 3.0, 4.0}));

    c.backward();
    // ∂c/∂a = 1/b = 1/3 for each element
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{1.0/3.0, 1.0/3.0, 1.0/3.0}));
    // ∂c/∂b = sum(-a/b²) = -(6+9+12)/9 = -3
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{-3.0}));
}

TEST(TensorDivision, DivisionByOne) {
    Tensor a({1.0, 2.0, 3.0}, true);
    Tensor b({1.0, 1.0, 1.0}, true);
    Tensor c = a / b;

    EXPECT_TRUE(c.equals_approx(Tensor{1.0, 2.0, 3.0}));

    c.backward();
    // ∂c/∂a = 1/b
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{1.0, 1.0, 1.0}));
    // ∂c/∂b = -a/b²
    EXPECT_TRUE(b.gradient->equals_approx(Tensor{-1.0, -2.0, -3.0}));
}

TEST(TensorDivision, DivisionByZeroThrows) {
    Tensor a({1.0}, true);
    Tensor b({0.0}, true);
    EXPECT_THROW(a / b, std::runtime_error);
}

TEST(TensorDivision, GradientWithBroadcastAndScalar) {
    Tensor a({{1.0, 2.0}, {3.0, 4.0}}, true);
    Tensor scalar({2.0}, true);
    
    Tensor c = a / scalar;
    c.backward();

    // Expected: ∂(a/s)/∂a = 1/s
    EXPECT_TRUE(a.gradient->equals_approx(Tensor{{1.0/2.0, 1.0/2.0}, 
                                                 {1.0/2.0, 1.0/2.0}}));

    // Expected: ∂(a/s)/∂s = -a/s²
    Tensor expected_grad_scalar = {-1.0/4.0 - 2.0/4.0 - 3.0/4.0 - 4.0/4.0};
    EXPECT_TRUE(scalar.gradient->equals_approx(expected_grad_scalar));
} 