![ember-logo](https://github.com/user-attachments/assets/bc80ea7a-e41c-4281-87b5-b9ba77ae317b)

# Ember🔥: Reverse-mode autodiff made easy

Ember is a small C++ library implementing reverse-mode automatic differentiation. 

Inspired by [micrograd](https://github.com/karpathy/micrograd), Ember offers a streamlined implementation of reverse mode automatic differentation (autodiff) without the complex optimizations found in production-grade frameworks. With an API that's _mostly_ similar to PyTorch's `Autograd` module, Ember aims to serve as both a reference implementation to help you grasp the core ideas of autodiff and a stepping stone for understanding the inner workings of modern neural network frameworks.


## Why Ember?

As one of the core techniques used in training modern neural networks, understanding autodiff is crucial. However, trying to understand this concept by reading the source code of a production-grade framework like PyTorch can be pretty challenging, with all its necessary optimizations. 

Ember aims to make this easier by providing a clear, concise, and well documented implementation of key concepts like computational graphs and backward functions, while keeping mostly the same interface as PyTorch.

Ultimately, Ember is designed to help you:
 - Build and (slowly) train neural networks.
 - Deepen your understanding of gradients, computational graphs, and backpropagation.
 - Seamlessly transition to understanding the inner workings of PyTorch (and other frameworks).
 - Learn the magic behind machine learning.

## Getting Started

### Prerequisites

- A C++17 compatible compiler (e.g., GCC, Clang)
- CMake for build automation
- A basic understanding of the following:
  - Neural networks
  - Partial derivatives
  - Gradient-based optimization
  - Computational graphs
  - Reverse-mode automatic differentiation

### Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/austinagii/Ember.git
   cd Ember
   ```
2. Build the project:  
   You can use the build script in the root directory to build the project.
   ```bash
   sh build.sh
   ```
3. (Optionally) Run tests:  
   ```bash
   ./build/ember_test
   ```

## Usage

Below is an example of how you can use Ember to create a function and calculate the partial derivatives of that function.

```cpp
#include <ember/tensor.h>

#include <iostream>

using namespace ember;

int main() {
    // Create tensors of different dimensions
    Tensor a({{1.0, 2.0}, {3.0, 4.0}}, true);  // true here indicates requires gradients to be computed
    Tensor b({{2.0, 1.0}, {0.5, 2.0}}, true);
    Tensor c({3.0}, true);

    // Perform element-wise operations with broadcasting
    auto d = (a * b) + c;  
    auto e = d / 2; 

    // Create a more complex computation graph.
    auto f = (d * a - b) / (c + 2);

    std::cout << "Output tensor:" << std::endl;
    std::cout << e.data << std::endl;

    // Calculate gradients using reverse-mode autodiff
    e.backward();

    std::cout << "\nGradients:" << std::endl;
    std::cout << "∂e/∂a = " << a.gradient.data << "\n\n";
    std::cout << "∂e/∂b = " << b.gradient.data << "\n\n";
    std::cout << "∂e/∂c = " << c.gradient.data << "\n\n";

    return 0;
}
```

Output:
```
Output tensor:
{{ 0.0625, 0.5000 },
 { 0.8333, 1.4286 }}

Gradients:
∂e/∂a = 
{{ 0.4219, 0.3750 },
 { 0.3444, 0.4745 }}

∂e/∂b = 
{{ -0.0703, 0.0000 },
 { 0.1333, 0.0918 }}

∂e/∂c = -0.0366
```

## Understanding The Code

To get started understanding the code, head to the `include` directory. Here you'll find a README that serves as a reading guide for the files you'll want to understand. 


## What's Next for Ember?

To ensure that Ember isn't just a toy, it needs to be capable of building a full, modern neural network. Therefore, the next step is to implement a moderately sized neural network from scratch using Ember. VGG16 is the current target for this.  

## 🌟 Join the Journey

I'm dedicated to making Ember a useful tool for learning and exploring neural networks. Any feedback and / or contributions are invaluable!
 - Have ideas? Open a discussion or submit an issue.
 - Want to contribute? Just go ahead, there's no contributing guide yet.
 - Stay updated! Follow the project to stay up to date.

# License

Ember is free and open source software licensed under the MIT License.
