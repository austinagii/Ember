# Project Logbook

## Log Entries

### January 11, 2025
- **Task**: Re-implement the arithmetic operations using the new Node and Tensor classes
- **Progress**:  
  - Simplify tensor usage by including tensor operations with tensor headers
- **Next Steps**:  
  - 

### January 10, 2025
- **Task**: Re-implement the arithmetic operations using the new Node and Tensor classes
- **Progress**:  
  - Added support for addition, subtraction and division
- **Next Steps**:  
  - Compare the current implementation with PyTorch's implementation

### January 7-8, 2025
- **Task**: Implement gradient accumulator based on PyTorch's autograd review
- **Progress**:  
  - Implemented gradient accumulator and tested with improved mul operator
- **Notes & Challenges**: 
  - Circular dependency between Node and Tensor structs caused some headaches
  - Returning an empty tensor vector from the gradient accumulator is a temporary solution
- **Next Steps**:  
  - Implement remaining arithmetic operations

### January 5-6, 2025
- **Task**: Review and implement core PyTorch autograd concepts
- **Progress**:  
  - Implemented Node and Tensor classes
  - Converted existing ops' backward functions to `Node`s
  - Started implementing autograd's `Engine` class
  - Studied torch::autograd::Engine::evaluate_function for gradient handling
- **Notes & Challenges**: 
  - Found [this discussion](https://dev-discuss.pytorch.org/t/how-to-read-the-autograd-codebase/383?utm_source=chatgpt.com) to be very helpful
  - Learned key concepts:
    - Leaf tensor gradients are captured in an accumulator
    - Internal function gradients are captured in a gradient buffer
- **Next Steps**:  
  - Implement gradient accumulator

### January 4, 2025
- **Task**: Add support for division and subtraction 
- **Progress**:  
  - Swapped out the named functions with operator overloads to simplify the interface. After reviewing PyTorch's implementation though this _may_ change.
  - Added support for subtraction.
  - Added support for division.
  - Started reviewing PyTorch's Autograd implementation for use as a reference. Specifically focused on [nodes](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/function.cpp), [edges](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/edge.h) and [variables](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.cpp).
- **Notes & Challenges**: 
  - PyTorch's Autograd implementation is fairly complex, taking longer to get a grasp on it than I would've hoped.
- **Next Steps**:  
  - Continue reviewing PyTorch Autograd implementation.
  - Cosolidate tests for add and multiply operations.
  - Add support for operations on tensors.

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
