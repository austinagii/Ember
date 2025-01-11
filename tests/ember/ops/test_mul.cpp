#include <ember/tensor.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace ember;

TEST(Ops, Multiplication) {
  Tensor a(3.0f);
  Tensor b(2.0f);
  Tensor c(4.0f);

  auto d = a * b * c;
  EXPECT_EQ(d.value, 24.0f);

  d.backward();
  EXPECT_EQ(d.gradient->value, 1.0f);
  EXPECT_EQ(c.gradient->value, 6.0f);
  EXPECT_EQ(a.gradient->value, 8.0f);
  EXPECT_EQ(b.gradient->value, 12.0f);
}
