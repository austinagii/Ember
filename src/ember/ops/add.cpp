#include <ember/ops/add.h>
#include <ember/tensor.h>
#include <ember/autograd/node.h>


namespace ember {

static Tensor add_tensors(Tensor& augend, Tensor& addend) {
  auto sum = Tensor(augend.value + addend.value);
  sum.gradient_fn = new AddBackward(augend, addend);
  return sum;
}

Tensor operator+(Tensor& augend, Tensor& addend) {
  return add_tensors(augend, addend);
}

Tensor operator+(Tensor& augend, Tensor&& addend) {
  return add_tensors(augend, addend);
}

Tensor operator+(Tensor&& augend, Tensor& addend) {
  return add_tensors(augend, addend);
}

Tensor operator+(Tensor&& augend, Tensor&& addend) {
  return add_tensors(augend, addend);
}

AddBackward::AddBackward(Tensor& augend, Tensor& addend) {
  edges.push_back(autograd::Edge(0, augend.get_gradient_edge()));
  edges.push_back(autograd::Edge(1, addend.get_gradient_edge()));
}

std::vector<Tensor> AddBackward::operator()(Tensor output_grad) {
  return {output_grad, output_grad};
}

}
