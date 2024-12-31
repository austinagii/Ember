#include <gtest/gtest.h>

#include "graph.h"

using namespace hyper;


TEST(Value, Addition) {
  auto a = new Value(3.0);
  auto b = new Value(2.0);

  auto c = Add(a, b);
  EXPECT_EQ(c.forward(), 5);
}
