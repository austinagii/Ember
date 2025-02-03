![ember-logo](https://github.com/user-attachments/assets/bc80ea7a-e41c-4281-87b5-b9ba77ae317b)

# Ember🔥: Reverse-mode autodiff made easy

Ember is a small C++ library implementing reverse-mode automatic differentiation. 

Inspired by [micrograd](https://github.com/karpathy/micrograd), Ember offers a streamlined implementation of reverse mode autodiff without the complex optimizations found in production grade frameworks. With an API that's (mostly) similar to PyTorch's `Autograd` module, Ember aims to serve as both a reference implementation to help you grasp the core ideas of autodiff and a stepping stone for understanding the inner workings of modern neural network frameworks.


## Why Ember?

As one of the core concepts used in training modern neural networks, understanding automatic differentation can help to facilitate a better intuition of neural networks. However, trying to understand this concept by reading the source code of a production-grade framework like PyTorch can be pretty challenging, with all it's necessary optimizations. 
If you're like me, you like to peek under the hood to get a good idea of how things work but wished you could just see the implementation of the core concepts without getting lost in the details.  

Ember aims to accomplish this by providing a clear, concise and documentation first approach to implementating key concepts like computational graphs and reverse-mode automatic differentiation, while keeping mostly the same interface as PyTorch.

Ember is designed to help you:
 - Build and (slowly) train neural networks.
 - Deepen your understanding of gradients, backpropagation, and loss functions.
 - Seamlessly transition to PyTorch with a similar interface and workflow.

Essentially, if you're a curious person and want to understand the "magic" behind machine learning, Ember is designed to make that easy. 


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

Before diving into the code, take a look at the below example. This example shows a (mathematical) function can be constructed by performing operations on Ember's `Tensor`s. We can then calculate the partial derivative of those functions with respect to it's inputs. 

#### Example

```cpp
#include <ember/tensor.h>
#include <iostream>

using namespace ember;

int main() {
    // Create tensors of different dimensions
    Tensor a({{1.0, 2.0}, {3.0, 4.0}}, true);
    Tensor b({{2.0, 1.0}, {0.5, 2.0}}, true);
    Tensor c({3.0}, true);

    // Perform element-wise operations with broadcasting
    auto d = (a * b) + c;  
    auto e = d / 2; 

    // Create a more complex computation graph
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

The main purpose of creating Ember was to develop a deeper understanding of automatic differentiation. It's one thing to be able to read and understand the concept, but a completely different thing when you want to see the internal of how it works.  

This is the order I'd recommend for having the smoothest time understanding the code. Let's say we have the following function:
$$f = x^2 + 2x + 1$$

Now, using Ember you can construct the function as follows:
```c++
#include <ember/tensor.h>
#include <iostream>

using namespace ember;

Tensor f(Tensor x) {
    return x * x + 2 * x + 1;
}

int main() {
    auto output = f(1);
    std::cout << output.data << std::endl;
    return 0;
}
```
In this example, we've created a function `f` that takes a `Tensor` and performs a series of operations on it. Specifically, multiplications and additions. When you think of a computational graph for this function, you may think of it like this:

```
               ┌────────────────────────────┐
               │ output = x * x + 2 * x + 1 │ 
               └────────────┬───────────────┘
                            │
                          ( + )  ← final addition (+)
                         /     \
                        /       1
                       /
                     ( + )      ← addition of x² and 2x
                    /     \
                   /       \
                ( * )     ( * )  ← multiplication nodes
                /   \     /   \
               x     x   2     x
```

This computational graph though, represents the computation of the output, which is not actually what we really need the computation graph for. What we actually want is a graph that looks like this:




1. `tensor.h` (`include/ember/tensor.h`)  
   `Tensor` is the heart of Ember. It is a thin wrapper around an [xtensor](https://xtensor.readthedocs.io/en/latest/) array, providing additional scaffolding for hooking into the computational graph. Once you've read the `Tensor` source, you'll understand what `Tensor`s represent, their role in the computational graph and how they are constructed. 
2. `node.h` (`include/ember/autograd/node.h`)  
   `Node` is the backbone of the computation graph. It is a base class for all nodes in the graph, and provides a common interface for all nodes.
3. `engine.h` (`include/ember/autograd/engine.h`)  
   `Engine` is the core of the autograd engine. It is responsible for constructing the computational graph, performing forward and backward passes, and computing gradients.

## What's Next for Ember?

To ensure that Ember isn't just a toy, it needs to be capable of building a full, modern neural network. Therefore, the next step is to implement a moderately sized neural network from scratch using Ember. VGG16 is the current target for this.  

## 🌟 Join the Journey

I'm dedicated to making Ember a useful tool for learning and exploring neural networks. Any feedback and / or contributions are invaluable!
 - Have ideas? Open a discussion or submit an issue.
 - Want to contribute? Just go ahead, there's no contributing guide yet.
 - Stay updated! Follow the project to stay up to date.

Let Ember spark your understanding of neural networks! 🔥
