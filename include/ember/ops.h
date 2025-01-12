// #include <ember/tensor.h>
//
// #include <gtest/gtest.h>
//
// using namespace ember;
//
// TEST(Ops, TensorOperationsOnConstants) {
//   auto a = Tensor(5.0f);
//   auto c = 5 * b;
//   c.backward();
//
//   EXPECT_EQ(a.grad, 5.0f);
//   EXPECT_EQ(b.grad, 5.0f);
// }
//
//
// TEST(Ops, TensorOperationsOnConstants) {
//   auto a = Tensor(5.0f);
//   auto b = Tensor(6.0f);
//   auto c = a * b;
//
//   auto d = Tensor
//   c.backward();
//
//   EXPECT_EQ(a.grad, 5.0f);
//   EXPECT_EQ(b.grad, 5.0f);
// }
