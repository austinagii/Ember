# Ember

## Overview
The `ember/include` directory contains the header files that form the core of 
the library.

These files provide a complete implementation of reverse mode automatic 
differentiation (also known as backpropagation) - the same algorithm used by 
modern deep learning frameworks like PyTorch and TensorFlow.

The codebase is surprisingly small and focused, containing just the essential 
components:
- A tensor data structure for storing and manipulating n-dimensional arrays
- Basic mathematical operations (add, multiply, etc.) with their corresponding
  gradient functions
- A computational graph system for tracking operations
- An engine for calculating gradients by traversing the computational graph

The structure of the directory is as follows:
```
ember/include/
├── autograd/  # the automatic differentiation implementation
│   ├── README.md
│   ├── node.h
│   ├── ...
├── ops/  # implementations of basic mathematical operations
│   ├── README.md
│   ├── add.h
│   ├── ...
├── tensor.h  # core tensor data structure and methods
```

This structure is mirrored in the `src` and `tests` directories, making it 
easy to find the corresponding implementation and tests for a given file.

Only one file is included in the top level directory: `tensor.h`. This 
file contains the core tensor data structure and the methods for performing 
operations on tensors. As such, it is the only file that needs to be included 
when using the library and the first file you'll need to read when getting 
started.

## Reading Guide
Expanding on the previous section, the following order is recommended for 
getting started reading the Ember source:

1. **tensor.h** (`include/ember/tensor.h`)  
   The core data structure in Ember. This file will show you how Ember 
   implements tensors and how they interact with the computational graph.

2. **ops/** (`include/ember/ops/`)  
   The `ops` folder contains implementations of basic mathematical operations 
   on tensors. At first, you only really need to read `add.h`. It's the 
   simplest and you'll be able to see how an operation is implemented. You'll 
   also see references to the `Node` struct. At this point, you can hop over 
   to the `autograd` directory.

3. **autograd/** (`include/ember/autograd/`)  
   `autograd` contains the implementation of automatic differentiation. With an
   understanding of `Tensor` and the add operation under your belt, you should
   have enough context to be able to read the code here.

Each subdirectory contains a README file that provides an overview of the files 
contained within the directory and how they fit into the overall structure of 
the library. As you navigate through the source, you can refer back to these.

## Usage 
The top level README contains a basic usage guide for the library. Refer to it 
for a quick overview of how to use a `Tensor`.

## Testing
As noted in the overview, the structure of the files in `include/ember/` is 
mirrored in the `src` and `tests` directories. Here you'll find the tests for 
the Tensor struct at `tests/ember/test_tensor.cpp`. 

Additionally, a `test_readme.cpp` file is provided to validate the examples in 
the README.

## Common Issues

1. **Shape Mismatch in Operations**  
   When performing operations between tensors of different shapes, ensure they are 
   broadcastable. Broadcasting follows NumPy-style rules.
   ```cpp
   Tensor a = {{1.0, 2.0}, {3.0, 4.0}};  // 2x2
   Tensor b = {1.0};                     // scalar (will broadcast)
   auto c = a + b;                       // valid
   ```

2. **Gradient Calculation**  
   - Ensure `backward()` is called on the final tensor in your computation
   - Intermediate gradients are not retained by default

3. **Memory Management**  
   - Tensors manage their own memory for gradients and computational graph nodes
   - When using temporary tensors in expressions, they are automatically cleaned up
   ```cpp
   auto c = a + Tensor({1.0});  // Temporary tensor is properly handled
   ```
