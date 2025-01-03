# HyperGrad: Neural Network Visualization and Training Library

HyperGrad is a C++ library designed to facilitate a better understanding of neural networks by enabling the creation, training, and visualization of their components. By building a directed acyclic graph (DAG) to represent networks as computational graphs, HyperGrad leverages reverse-mode automatic differentiation to compute gradients efficiently.

---

## Features

- **Dynamic Computational Graph**: Construct neural networks as a DAG.
- **Automatic Differentiation**: Compute gradients of the loss with respect to any input node.
- **Neural Network Training**: Visualize and train neural networks with a focus on learning mechanics.
- **Extensible Operations**: Support for basic mathematical operations, with plans for tensor support.

---

## Getting Started

### Prerequisites

- A C++17 compatible compiler (e.g., GCC, Clang)
- CMake for build automation
- A basic understanding of neural networks and gradient-based optimization

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/austinagii/hypergrad.git
   cd HyperGrad
   ```
2. Build the project: </br>
   You can use the build script in the root directory to build the project.
   ```bash
   sh build.sh
   ```
3. Run tests: </br>
   ```bash
   ./build/hypergrad_test
   ```

### Usage

#### Example

```cpp
nclude "graph.h"

int main() {
    // Create nodes
    Node a(5.0); 
    Node b(3.0);

    // Perform operations
    Node c = multiply(a, b);

    // Backpropagate to compute gradients
    c.backward(); 
    
    // Access gradients
    std::cout << "Gradient of a: " << a.grad << std::endl;

    return 0;
}
```
