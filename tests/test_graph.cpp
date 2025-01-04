#include <gtest/gtest.h>

#include "graph.h"

using namespace hyper;


TEST(Value, Addition) {
  auto a = Node(3.0);
  auto b = Node(2.0);
  auto c = a + b;

  EXPECT_EQ(c.value, 5);
}

TEST(Node, BackpropagateThroughAdditionOperation) {
  auto a = Node(3.0);
  auto b = Node(2.0);
  auto c = a + b;
  c.compute_gradient(1.0);

  EXPECT_EQ(a.gradient, 1.0);
  EXPECT_EQ(b.gradient, 1.0);
  EXPECT_EQ(c.gradient, 1.0);
}

TEST(Node, BackpropagateThroughNestedAdditionOperation) {
  auto a = Node(3);
  auto b = Node(2);
  auto c = a + b;
  auto d = Node(5);
  auto e = c + d;

  e.compute_gradient(1.0);

  EXPECT_EQ(a.gradient, 1.0);
  EXPECT_EQ(b.gradient, 1.0);
  EXPECT_EQ(c.gradient, 1.0);
  EXPECT_EQ(d.gradient, 1.0);
  EXPECT_EQ(e.gradient, 1.0);
}

TEST(Node, Subtraction) {
  auto a = Node(3);
  auto b = Node(2);
  auto c = a - b;
  EXPECT_EQ(c.value, 1);

  auto d = Node(5);
  auto e = c - d;
  EXPECT_EQ(e.value, -4);

  e.compute_gradient(1.1);
  EXPECT_EQ(a.gradient, 1.1f);
  EXPECT_EQ(b.gradient, 1.1f);
  EXPECT_EQ(c.gradient, 1.1f);
  EXPECT_EQ(d.gradient, 1.1f);
  EXPECT_EQ(e.gradient, 1.1f);
}

TEST(Value, Multiplication) {
  auto a = Node(3.0f);
  auto b = Node(2.0f);

  auto c = a * b;

  EXPECT_EQ(c.value, 6.0f);
}

TEST(Node, BackpropagateThroughMultiplicationOperation) {
  auto a = Node(3.0f);
  auto b = Node(2.0f);
  auto c = a * b;

  c.compute_gradient(1.0f);

  EXPECT_EQ(a.gradient, 2.0f);
  EXPECT_EQ(b.gradient, 3.0f);
  EXPECT_EQ(c.gradient, 1.0f);
}

TEST(Node, BackpropagateThroughNestedMultiplicationOperation) {
  auto a = Node(3);
  auto b = Node(2);
  auto c = a * b;
  auto d = Node(8);
  auto e = c * d;

  e.compute_gradient(2);

  EXPECT_EQ(a.gradient, 32);
  EXPECT_EQ(b.gradient, 48);
  EXPECT_EQ(c.gradient, 16);
  EXPECT_EQ(d.gradient, 12);
  EXPECT_EQ(e.gradient, 2);
}

TEST(Node, BackpropagateThroughMultiplicationOperationHard) {
  auto a = Node(2.0f);
  auto b = Node(3.0f);
  auto c = a * b;
  auto d = Node(4.0f);
  auto e = c + d;
  auto f = Node(2.0f);
  auto g = e * f;
  auto h = Node(5.0f);
  auto i = g + h;

  i.compute_gradient(0.5);

  EXPECT_EQ(i.gradient, 0.5f);
  EXPECT_EQ(h.gradient, 0.5f);
  EXPECT_EQ(g.gradient, 0.5f);
  EXPECT_EQ(f.gradient, 5.0f);
  EXPECT_EQ(e.gradient, 1.0f);
  EXPECT_EQ(d.gradient, 1.0f);
  EXPECT_EQ(c.gradient, 1.0f);
  EXPECT_EQ(b.gradient, 2.0f);
  EXPECT_EQ(a.gradient, 3.0f);
}
