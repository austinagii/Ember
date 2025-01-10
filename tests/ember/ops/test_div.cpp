#include <ember/tensor.h>
#include <ember/ops/div.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace ember;

TEST(Ops, Division) {
    Tensor a(24.0f);
    Tensor b(2.0f);
    Tensor c(3.0f);

    auto d = a / b / c;
    EXPECT_EQ(d.value, 4.0f);

    d.backward();
    EXPECT_EQ(d.gradient->value, 1.0f);
    EXPECT_EQ(c.gradient->value, -12.0f / 9.0f);
    EXPECT_EQ(a.gradient->value, 1.0f / 6.0f);
    EXPECT_EQ(b.gradient->value, -2.0f);
} 
