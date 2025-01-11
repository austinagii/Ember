#include <ember/tensor.h>

#include <gtest/gtest.h>

using namespace ember;

TEST(Tensor, CreateTensorCheckpoint) {
  Tensor t(10.0f);
  TensorSnapshot c = t.save();
  
  EXPECT_EQ(c.value, t.value);
}
