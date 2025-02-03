#include <ember/tensor.h>
#include <gtest/gtest.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(ReadmeExample, ComputationAndGradientsAreCorrect) {
    // Create tensors
    Tensor a({{1.0, 2.0}, {3.0, 4.0}}, true);
    Tensor b({{2.0, 1.0}, {0.5, 2.0}}, true);
    Tensor scalar({3.0}, true);

    // Perform element-wise operations with broadcasting
    Tensor c = (a * b) + scalar;
    Tensor d = c / 2;
    Tensor e = (d * a - b) / (c + scalar);

    // Expected results for d and e
    Tensor expected_result_d = {{2.5, 2.5}, {2.25, 5.5}};
    Tensor expected_result_e = {{0.0625, 0.5}, {0.8333333, 1.4285714}};

    // Compute gradients through backpropagation
    e.backward();

    // Expected gradients when backpropagating from e
    Tensor expected_grad_a = {{0.421875, 0.375}, {0.3444445, 0.4744898}};
    Tensor expected_grad_b = {{-0.0703125,  0.0000000}, {0.1333334,  0.0918367}};
    Tensor expected_grad_scalar = {-0.0365717};

    EXPECT_TRUE(d.equals_approx(expected_result_d));
    EXPECT_TRUE(e.equals_approx(expected_result_e));
    EXPECT_TRUE(a.gradient->equals_approx(expected_grad_a));
    EXPECT_TRUE(b.gradient->equals_approx(expected_grad_b));
    EXPECT_TRUE(scalar.gradient->equals_approx(expected_grad_scalar));
}