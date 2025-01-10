#include <ember/autograd/node.h>
#include <ember/tensor.h>

#include <vector>


namespace ember {

struct AddBackward;

/**
 * Add two tensors and return a new tensor representing the sum.
 */
Tensor operator+(Tensor& augend, Tensor& addend); 
Tensor operator+(Tensor& augend, Tensor&& addend); 
Tensor operator+(Tensor&& augend, Tensor& addend); 
Tensor operator+(Tensor&& augend, Tensor&& addend); 

struct AddBackward: public autograd::Node {

  AddBackward(Tensor& augend, Tensor& addend);
  virtual std::vector<Tensor> operator()(Tensor output_grad); 

};

} // namespace ember::ops
