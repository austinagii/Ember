#ifndef GRAPH_H
#define GRAPH_H 

#include <vector>
#include <functional>
#include <iostream>
#include <memory>

namespace hyper {

/**
 * Represents a generic operation in a computation.
 */
struct Node {
  // The current value of this node;
  float value;
  // The gradient of this node w.r.t. the ancestor on which backward was called.
  float gradient;
  // A callable which outputs gadient of this node when given the gradient of it's parent.
  std::function<float(float)> grad_fn;
  // The nodes that were used in the construction of this node.
  std::vector<std::reference_wrapper<Node>> children;

  Node(): value(0.0f), gradient(0.0f) {}

  Node(float value): value(value), gradient(0.0f) {}

  /**
   * Propagates the gradient back from the parent Node to this node.
   * N.B. This is not the typical backprop algorithm, since it does
   * not propagate the gradient at this node to it's children.
   */
  void compute_gradient(float gradient) {
    // Since gradient functions are assigned to nodes based on the operations they
    // are participate in, a function missing a gradient function is likely the last
    // operation in the tree and should just return 1.
    if(this->grad_fn == nullptr) {
      this->grad_fn = [](float v) { return v; };
    }
    this->gradient = this->grad_fn(gradient);

    for (auto iter = this->children.begin(); iter != this->children.end(); ++iter) {
      (*iter).get().compute_gradient(this->gradient);
    } 
  }
};

/**
 * Add two nodes and return a new node representing the sum.
 */
Node operator+(Node& augend, Node& addend) {
  auto sum = Node(augend.value + addend.value);
  sum.children.push_back(std::ref(augend));
  sum.children.push_back(std::ref(addend));

  augend.grad_fn = [](float v) { return v; };
  addend.grad_fn = [](float v) { return v; };
  return sum;
}

/**
 * Subtract one node from another and return a new node representing the difference.
 */
Node operator-(Node& minuend, Node& subtrahend) {
  auto sum = Node(minuend.value - subtrahend.value);
  sum.children.push_back(std::ref(minuend));
  sum.children.push_back(std::ref(subtrahend));

  minuend.grad_fn = [](float v) { return v; };
  subtrahend.grad_fn = [](float v) { return v; };
  return sum;
}

/**
 * Multiply two nodes and return a new node representing the product.
 */
Node operator*(Node& multiplicand, Node& multiplier) {
  auto product = Node(multiplicand.value * multiplier.value);
  product.children.push_back(std::ref(multiplicand));
  product.children.push_back(std::ref(multiplier));

  multiplicand.grad_fn = [&multiplier](float v) { return v * multiplier.value; };
  multiplier.grad_fn = [&multiplicand](float v) { return v * multiplicand.value; };
  return product;
}

/**
 * Divides on node by another and return a node representing the quotient.
 */
Node operator/(Node& dividend, Node& divisor) {
  auto quotient = Node(dividend.value / divisor.value);
  quotient.children.push_back(std::ref(dividend));
  quotient.children.push_back(std::ref(divisor));

  dividend.grad_fn = [&divisor](float v) { return v / divisor.value; };
  divisor.grad_fn = [&dividend, &divisor](float v) { return v * ((-dividend.value) / (divisor.value * divisor.value)); };
  return quotient;
}

}
#endif // GRAPH_H
