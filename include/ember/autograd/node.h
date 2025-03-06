#ifndef EMBER_AUTOGRAD_NODE_H
#define EMBER_AUTOGRAD_NODE_H

#include <ember/autograd/context.h>
#include <ember/autograd/edge.h>
#include <ember/tensor_snapshot.h>

#include <utility>
#include <vector>

namespace ember {
struct Tensor;
}

namespace ember::autograd {

/**
 * `Node` represents a vertex in the Directed Acyclic Graph (DAG) that forms the
 * computational graph for automatic differentiation.
 *
 * Along with `Edge`s, `Node`s are the fundamental building blocks of
 * computational graphs. Each Node represents a derivative function that
 * describes how partial derivatives are calculated for the inputs of a given
 * differentiable operation.
 *
 * For example, consider a differentiable function called 'spiral' that takes
 * two inputs `a` and `b` and returns an output `c`. The corresponding `Node`
 * for this function represents the gradient function that computes the partial
 * derivatives ∂c/∂a and ∂c/∂b. These derivatives indicate how much the output
 * `c` changes with respect to small changes in `a` and `b`.
 *
 * The most important component of a Node is its call operator (operator())
 * which must be overridden with an implementation that calculates the input
 * gradients for the corresponding forward operation.
 *
 * During the backward pass, a Node may *optionally* use a collection of
 * TensorSnapshots that contain values captured during the forward pass. The
 * number of calculated input gradients must have a 1:1 correspondence with the
 * number of input connections represented by the `edges` vector.
 */
struct Node {
  Node() = default;

  template <typename... Tensors>
  Node(Context ctx, Tensors&... inputs) : ctx(ctx) {
    std::size_t input_ix = 0;

    auto add_input = [this, &input_ix](auto& tensor) {
      if (tensor.requires_grad) {
        add_next_edge(autograd::Edge(input_ix, tensor.get_gradient_edge()));
      }
      saved_tensors.emplace_back(tensor.save());
      input_ix += 1;
    };
    (add_input(inputs), ...);
  }

  Context ctx;

  /**
   * @brief Vector of edges connecting this node to its input nodes in the
   * computational graph.
   *
   * Each edge represents a connection to an input tensor's gradient
   * computation. The order of edges corresponds to the order of input gradients
   * that will be computed in the operator().
   */
  std::vector<Edge> edges;

  /**
   * @brief Collection of tensor values captured during the forward pass.
   *
   * These snapshots can be used during the backward pass to compute accurate
   * gradients for operations that need access to the original input values
   * (e.g., division, power operations, etc.).
   */
  std::vector<TensorSnapshot> saved_tensors;

  /**
   * @brief Computes the gradients of each input with respect to the output
   * gradient for this operation.
   *
   * @param output_grad The gradient of the loss with respect to this node's
   * output
   * @return A vector of tensors containing the gradients for each input
   *
   * Example: For a binary addition operation (z = x + y), the corresponding
   * node returns gradients of [1, 1] since ∂z/∂x = 1 and ∂z/∂y = 1, meaning the
   * change in the output is directly proportional to changes in either input.
   */
  virtual std::vector<ember::Tensor> operator()(ember::Tensor output_grad) = 0;

  virtual ~Node() = default;

  /**
   * @brief Add a new input connection for this node.
   *
   * @param e The edge to add, representing a connection to an input tensor
   *
   * This method is typically called during the construction of the
   * computational graph to establish connections between operations.
   */
  void add_next_edge(Edge e) { edges.emplace_back(std::move(e)); }

  /**
   * @brief Return the number of input connections this node has.
   *
   * This count corresponds to the number of input gradients that will be
   * computed during the backward pass.
   */
  std::size_t get_num_inputs() { return edges.size(); }
};

}  // namespace ember::autograd

#endif  // EMBER_AUTOGRAD_NODE_H
