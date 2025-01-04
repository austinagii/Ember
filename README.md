![image](https://github.com/user-attachments/assets/28018382-3bc6-4260-ae65-1714bf689360)

# Ember🔥: A Simplified Autograd Library for Learning Neural Networks


Ember is a lightweight C++/Python library designed to help anyone (including me 😅) have a better understanding of the inner workings of neural networks and automatic differentiation. Inspired by [PyTorch](https://github.com/pytorch/pytorch)'s autograd engine with a mix of [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad) (I'd encourage you to check out the last two if you haven't already), Ember offers a simplified implementation while maintaining a familiar interface, making it a useful as a stepping stone to transitioning to PyTorch’s full capabilities.


## Why Ember?

Understanding the internals of a neural network’s training process can be pretty challenging, especially when diving right in with a production-grade framework like PyTorch, with all it's necessary optimizations. If you're like me, you like to peek under the hood to get a good idea of how things work but wished you could just see the implementation of the core concepts without getting lost in the details. 
</br></br>
Ember aims to accomplish this by providing a clear, concise and documentation first approach to implementating key concepts like computational graphs and reverse-mode automatic differentiation, while keeping mostly the same interface as PyTorch.

With Ember, you can:
 - Build and train **small** neural networks from scratch. (remember the lack of optimization?)
 - Visualize and interact with the computational graph. (being small also helps with this)
 - Deepen your understanding of gradients, backpropagation, and loss functions.
 - Seamlessly transition to PyTorch with a similar interface and workflow.

If you're a curious person and want to understand the “magic” behind machine learning, Ember is designed to make that easy. 


## Getting Started

### Prerequisites

- A C++17 compatible compiler (e.g., GCC, Clang)
- CMake for build automation
- A basic understanding of neural networks and gradient-based optimization

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/austinagii/Ember.git
   cd Ember
   ```
2. Build the project: </br>
   You can use the build script in the root directory to build the project.
   ```bash
   sh build.sh
   ```
3. (Optionally) Run tests: </br>
   ```bash
   ./build/ember_test
   ```

### Usage

#### Example

```cpp
nclude "ember.h"

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

## What’s Next for Ember?

We’re just getting started! Here’s a sneak peek at what’s on the roadmap for Ember:

### 🚀 Upcoming Features

#### Tensor Support

Ember will soon support tensor operations, enabling more complex and scalable neural network architectures.

#### Advanced Visualizations

Interactive and detailed visualizations of computational graphs, gradient flows, and training progress using d3.js.

#### Additional Neural Network Components

Implementations for additional mathemativeal operations, layers, activation functions, optimizers, and loss functions to accelerate experimentation.

#### Improved Documentation

Deep dives into the theory behind autograd, tutorials for building neural networks, and comparisons with frameworks like PyTorch (likely as a wiki here or at my website https://kadeemaustin.ai) 

#### Python API

An optional Python wrapper for those who prefer Python’s flexibility while maintaining Ember’s core functionality.


## 🌟 Join the Journey

I'm dedicated to making Ember a useful tool for learning and exploring neural networks. Any feedback and / or contributions are invaluable!
 - Have ideas? Open a discussion or submit an issue.
 - Want to contribute? Just go ahead, there's no contributing guide yet.
 - Stay updated! Follow the project to stay up to date.

Let Ember spark your understanding of neural networks! 🔥 (Yeah, pretty cheesy right? :D)
