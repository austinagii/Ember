#include "ember/ops/matmul.h"

#include "gtest/gtest.h"

using namespace ember;

TEST(TensorDot, DotProductIsCorrectlyCalculated) {
  Tensor a({{2.0, 3.0}, {5.0, 3.0}}, true);
  Tensor b({{5.0, 8.0}, {9.0, 2.0}}, true);

  Tensor c = ember::matmul(a, b);
  EXPECT_EQ(c, Tensor({{37.0, 22.0}, {52.0, 46.0}}));

  c.backward();

  ASSERT_TRUE(c.requires_grad);

  ASSERT_TRUE(a.requires_grad && a.gradient != nullptr);
  EXPECT_TRUE(a.gradient->equals_approx(Tensor({{13.0, 11.0}, {13.0, 11.0}})));

  ASSERT_TRUE(b.requires_grad && b.gradient != nullptr);
  EXPECT_TRUE(b.gradient->equals_approx(Tensor({{7.0, 7.0}, {6.0, 6.0}})));
}
