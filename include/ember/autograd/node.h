#ifndef NODE_H
#define NODE_H 

#include <ember/autograd/edge.h>
#include <ember/tensor_snapshot.h>

#include <utility>
#include <vector>

namespace ember {
    struct Tensor;  // Only need forward declaration for Tensor now
}

namespace ember::autograd {

struct Node {
    std::vector<Edge> edges;
    std::vector<TensorSnapshot> saved_tensors;
    virtual std::vector<ember::Tensor> operator()(ember::Tensor output_grad) = 0;
    virtual ~Node() = default;

  void add_next_edge(Edge e) {
    edges.emplace_back(std::move(e));
  }

  std::size_t get_num_inputs() {
    return edges.size();
  }
};

} // namespace ember::autograd

#endif // NODE_H
