#ifndef GRAPH_H
#define GRAPH_H 

#include <vector>

namespace hyper {

class Node {
  public:
  virtual float forward() = 0;
  // virtual Node backward() = 0;
  Node() {}
  virtual ~Node() {}

  protected:
  std::vector<Node*> children;
  float value;
};
  
/**
 * Represents a mathematical operation performed on one or more values.
 */
class Add: public Node {
  public:

  Add(Node* augend, Node* addend) {
    children.push_back(augend);
    children.push_back(addend);
  }
 
  float forward() {
    float sum = 0.0f;
    for(auto iter = children.begin(); iter != children.end(); ++iter) {
      sum += (*iter)->forward();
    }
    return sum;
  }
};

/**
 * A container for a floating point value.
 */
class Value: public Node {
  public:  
  Value(const float& v) {
    value = v;
  }

  float forward() {
    return value;
  }
};

}

#endif // GRAPH_H
