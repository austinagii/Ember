#ifndef EMBER_AUTOGRAD_EDGE_H
#define EMBER_AUTOGRAD_EDGE_H

#include <cstddef>

namespace ember::autograd {

// Forward declaration
struct Node;
 
struct Edge {
  Node* fn;
  std::size_t input_nr;

  Edge(std::size_t input_nr, Node* fn): input_nr(input_nr), fn(fn) {}
}; // struct Edge

} // namespace ember::autograd

#endif // !EMBER_AUTOGRAD_EDGE_H