#include <ember/ops/exp.h>

#include <gtest/gtest.h>

using namespace ember;

TEST(TensorExponentiation, ExponentIsCorrectlyComputed) {
  Tensor a({4.0, 3.0}, true);

  Tensor actual_result = a.exp();

  Tensor expected_result({54.59815, 20.08553});

  EXPECT_TRUE(actual_result.equals_approx(expected_result));

  actual_result.backward();

  ASSERT_TRUE(a.requires_grad() && a.gradient != nullptr);
  EXPECT_TRUE(a.gradient->equals_approx(expected_result));
}

TEST(TensorExponentiation, ExponentIsCorrectlyComputedForNestedOperations) {
  Tensor a({{0.2, 0.8}, {0.5, 1.1}}, true);

  Tensor b = ember::exp(a);

  EXPECT_TRUE(b.equals_approx(
      Tensor({{1.221402758, 2.225540928}, {1.648721271, 3.004166024}})));

  Tensor c = ember::exp(b);

  EXPECT_TRUE(c.equals_approx(
      Tensor({{3.391942472, 9.258509299}, {5.200302570, 20.169304633}})));

  c.backward();

  ASSERT_TRUE(a.requires_grad() && a.gradient != nullptr);
  EXPECT_TRUE(a.gradient->equals_approx(
      Tensor({{4.14292789, 20.60519138}, {8.573849463, 60.59193971}})));
}

TEST(TensorExponentiation,
     ExponentIsCorrectlyComputedForAnonymousIntermediateTensor) {
  Tensor a({{0.2, 0.8}, {0.5, 1.1}}, true);

  Tensor actual_result = ember::exp(ember::exp(a));
  Tensor expected_result(
      {{3.391942472, 9.258509299}, {5.200302570, 20.169304633}});

  EXPECT_TRUE(actual_result.equals_approx(expected_result));

  actual_result.backward();

  Tensor expected_gradient(
      {{4.14292789, 20.60519138}, {8.573849463, 60.59193971}});

  ASSERT_TRUE(a.requires_grad() && a.gradient != nullptr);
  EXPECT_TRUE(a.gradient->equals_approx(expected_gradient));
}
