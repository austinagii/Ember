#include <ember/tensor.h>
#include <gtest/gtest.h>
#include <xtensor/xio.hpp>
#include <iostream>

using namespace ember;

TEST(ReadmeExample, ComputationAndGradientsAreCorrect) {
    // Create tensors
    Tensor a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor b = {{2.0f, 1.0f}, {0.5f, 2.0f}};
    Tensor scalar = 3.0f;

    // Perform element-wise operations with broadcasting
    auto c = (a * b) + scalar;
    auto d = c / 2;
    auto e = (d * a - b) / (c + scalar); // Uncommented calculation for e

    // Expected results for d and e
    xt::xarray<float> expected_result_d = {
        {2.5f, 2.5f},
        {2.25f, 5.5f}
    };

    xt::xarray<float> expected_result_e = {
        {0.0625f, 0.5f},
        {0.8333333f, 1.4285714f}
    };

    // Compute gradients through backpropagation
    e.backward();

    // Expected gradients when backpropagating from e
    xt::xarray<float> expected_grad_a = {
        {0.4218750f, 0.3750000f},
        {0.3444445f, 0.4744898f}
    };

    xt::xarray<float> expected_grad_b = {
        {-0.0703125f,  0.0000000f},
        {0.1333334f,  0.0918367f}
    };

    xt::xarray<float> expected_grad_scalar = {-0.0365717};

    // Assertions
    EXPECT_TRUE(xt::allclose(d.data, expected_result_d, 0.001f)) << "Mismatch in tensor d values.";
    EXPECT_TRUE(xt::allclose(e.data, expected_result_e, 0.001f)) << "Mismatch in tensor e values.";
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a, 0.001f)) << "Mismatch in gradients for tensor a.";
    EXPECT_TRUE(xt::allclose(b.gradient->data, expected_grad_b, 0.001f)) << "Mismatch in gradients for tensor b.";
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, expected_grad_scalar, 0.001f)) << "Mismatch in gradients for scalar.";
}
