#include <ember/tensor.h>
#include <ember/ops/sub.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace ember;

TEST(Ops, Subtraction) {
    Tensor a(10.0f);
    Tensor b(3.0f);
    Tensor c(2.0f);

    auto d = a - b - c;
    EXPECT_EQ(d.value, 5.0f);

    d.backward();
    EXPECT_EQ(d.gradient->value, 1.0f);
    EXPECT_EQ(c.gradient->value, -1.0f);
    EXPECT_EQ(a.gradient->value, 1.0f);
    EXPECT_EQ(b.gradient->value, -1.0f);
} 