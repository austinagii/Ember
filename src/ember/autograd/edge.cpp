#include "ember/autograd/edge.h"

namespace ember::autograd {

Edge::Edge(std::size_t input_nr, Node* fn)
    : input_nr(input_nr)
    , fn(fn) {}

} // namespace ember::autograd