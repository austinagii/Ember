#include <ember/ops/exp.h>

#include <gtest/gtest.h>

using namespace ember;

TEST(TensorExponentiation, ExponentIsCorrectlyComputed) {
  Tensor a({4.0, 3.0}, true);

  Tensor actual_result = exp(a);

  Tensor expected_result({54.59815, 20.08553});

  EXPECT_TRUE(actual_result.equals_approx(expected_result));

  actual_result.backward();

  ASSERT_TRUE(a.requires_grad && a.gradient != nullptr);
  EXPECT_TRUE(a.gradient->equals_approx(expected_result));
}
