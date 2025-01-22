#include <ember/tensor.h>
#include <gtest/gtest.h>
#include <xtensor/xio.hpp>

using namespace ember;

TEST(ReadmeExample, ComputationAndGradientsAreCorrect) {
    // Create tensors exactly as shown in README
    Tensor a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor b = {{2.0f, 1.0f}, {0.5f, 2.0f}};
    Tensor scalar = {3.0f};

    // Expected intermediate results
    xt::xarray<float> expected_c = {
        {5.0f, 5.0f},
        {4.5f, 11.0f}
    };

    xt::xarray<float> expected_d = {
        {2.5f, 2.5f},
        {2.25f, 5.5f}
    };

    xt::xarray<float> expected_result = {
        {0.0625f, 0.5000f},
        {0.8333f, 1.4290f}
    };

    // Expected gradients
    xt::xarray<float> expected_grad_a = {
        {0.3125f, 0.3125f},
        {0.3000f, 0.3929f}
    };

    xt::xarray<float> expected_grad_b = {
        {-0.1250f, -0.1250f},
        {-0.1333f, -0.0714f}
    };

    xt::xarray<float> expected_grad_scalar = {-0.2857f};

    // Perform operations as shown in README and print intermediate results
    auto c = (a * b) + scalar;
    std::cout << "\nStep 1:";
    std::cout << "\na * b = " << (a * b).data;
    std::cout << "\nc = (a * b) + scalar = " << c.data;

    auto d = c / Tensor({2.0f});
    std::cout << "\n\nStep 2:";
    std::cout << "\nd = c/2 = " << d.data;

    auto e = (d * a - b) / (c + scalar);
    std::cout << "\n\nStep 3:";
    std::cout << "\nd * a = " << (d * a).data;
    std::cout << "\nd * a - b = " << (d * a - b).data;
    std::cout << "\nc + scalar = " << (c + scalar).data;
    std::cout << "\ne = (d * a - b)/(c + scalar) = " << e.data;

    // Compute gradients
    e.backward();

    // Print expected vs actual gradients
    std::cout << "\n\nGradients:";
    std::cout << "\nExpected gradient for a:\n" << expected_grad_a;
    std::cout << "\nActual gradient for a:\n" << a.gradient->data;
    
    std::cout << "\n\nExpected gradient for b:\n" << expected_grad_b;
    std::cout << "\nActual gradient for b:\n" << b.gradient->data;
    
    std::cout << "\n\nExpected gradient for scalar:\n" << expected_grad_scalar;
    std::cout << "\nActual gradient for scalar:\n" << scalar.gradient->data;

    // Original assertions
    EXPECT_TRUE(xt::allclose(e.data, expected_result, 0.001f));
    EXPECT_TRUE(xt::allclose(a.gradient->data, expected_grad_a, 0.001f));
    EXPECT_TRUE(xt::allclose(b.gradient->data, expected_grad_b, 0.001f));
    EXPECT_TRUE(xt::allclose(scalar.gradient->data, expected_grad_scalar, 0.001f));

    // Verify intermediate results
    EXPECT_TRUE(xt::allclose(c.data, expected_c, 0.001f));
    EXPECT_TRUE(xt::allclose(d.data, expected_d, 0.001f));
} 