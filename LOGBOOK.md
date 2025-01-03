# Project Logbook: HyperGrad

## Project Overview
**Goal**: To create a fully functional library for building, training and visualizing neural networks. </br>
**Applications**: Mostly for facilitating a better understanding of neural networks and how they are trained. </br>
**Approach**: Builds a DAG representing the network as a computational graph and uses reverse mode automatic differentiation to compute the gradient of the loss w.r.t to each input.

## Log Entries

### January 2, 2025
- **Task**: Add support for multiplication and improve gradient calculation.
- **Progress**:  
  - Added the `mul` method to calculate the product of two nodes.
  - Added a new member variable `grad_fn` on each node that is defined during the graph's construction and can be used to calculate the gradient at a given node, given it's parent's gradient
- **Notes & Challenges**:  
  - Encountered some errors calculating the gradients due to assigning the gradient function after a duplicate child node was unintentionally created during node graph construction.
- **Next Steps**:  
  - Add support for division and subtraction 
  - Add support for operations on tensors

### January 1, 2025
- **Task**: Compute the gradient for each node in a given graph.
- **Progress**:  
  - Simplified the original classes to just one `Node` class which stores the value, the gradient and any children
  - Implemented the recursive gradient calculation from the root to leaf nodes in the graph
- **Notes & Challenges**:  
  - Trying to maintain different subclasses of `Node` and determining how to implment the gradient calculation for each of them was more comlicated than it needed to be.
- **Next Steps**:  
  - Add support for multiplication 
  - Improve the gradient calculation implementation

### Dec 31, 2025
- **Task**: Create a computational graph from addition operations and compute the result.
- **Progress**:  
  - Implemented `Value` and `Operation` classes which inherit from a base `Node` class with basic support for addition 
- **Next Steps**:  
  - Add a function to compute the gradient at each node in the graph w.r.t. some ancestor
---
