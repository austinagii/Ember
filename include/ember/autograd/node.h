#ifndef NODE_H
#define NODE_H 

#include <vector>

namespace ember {
    struct Tensor;  // Forward declaration only
}

namespace ember::autograd {

struct Node {
    std::vector<Node*> next_fns;
    virtual std::vector<ember::Tensor> operator()(ember::Tensor output_grad) = 0;
    virtual ~Node() = default;  // Add virtual destructor for proper cleanup
};

} // namespace ember::autograd

#endif // NODE_H
