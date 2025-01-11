#ifndef NODE_H
#define NODE_H 

#include <vector>
#include <ember/tensor_snapshot.h>

namespace ember {
    struct Tensor;  // Only need forward declaration for Tensor now
}

namespace ember::autograd {

struct Node {
    std::vector<Node*> next_fns;
    std::vector<TensorSnapshot> saved_tensors;
    virtual std::vector<ember::Tensor> operator()(ember::Tensor output_grad) = 0;
    virtual ~Node() = default;
};

} // namespace ember::autograd

#endif // NODE_H
