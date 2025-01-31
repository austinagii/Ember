# Autograd: Performing Automatic Differentiation Using Computational Graphs

## Overview

The `autograd` folder (named after PyTorch's automatic differentiation library)
contains the code that describes how computational graphs are constructed and 
used to perform automatic differentiation.  

The computational graphs that are constructed represent the reverse pass (gradient calculations) 
rather than the forward pass (function execution steps). This graph defines how 
gradients flow backward through the mathematical operations to compute derivatives.

Take the simple function $y = a \times b + c$. On the left hand side of the diagram 
below is the computational graph that represents the forward pass execution of
this function. In this graph, information flows from bottom to top starting with 
the inputs / leaf nodes ($a$, $b$, $c$), passing through each operation 
($\times$, $+$) and eventually terminating in the output / root node ($y$). 
This is fine for representing how the function calculates its output given a 
set of inputs, but what we want to know is how do slight changes in these inputs 
actually affect the output. To do this we create the computational graph on the 
right that represents the reverse pass. This graph is mostly the same as the 
forward pass in terms of structure, but with the direction of the edges reversed 
and the forward pass operations replaced with their corresponding gradient operations. 

```
            Forward Pass (Computation)             │             Backward Pass (Gradients)
           ----------------------------            │            ---------------------------
                                                   │    
               ┌───────────────┐                   │               ┌──────────────────┐
               │ y = a + b * c │                   │               │ ∂(y = a + b * c) │
               └──────┬────────┘                   │               └─────────┬────────┘
                      │                            │                         │
                    ( + )                          │                      ∂( + )
                      │                            │                         │
                ┌─────┴─────┐                      │             ┌───────────┴───────────┐
                │           │                      │             │                       │
                │           │                      │             │                       │
              ┌─┴─┐         │                      │          ┌──┴───┐                   │
              │ a │       ( * )                    │          │ ∂(y) │                   │
              └───┘         │                      │          │ ──── │                ∂( * ) 
                      ┌─────┴─────┐                │          │ ∂(a) │                   │ 
                      │           │                │          └──────┘           ┌───────┴───────┐
                      │           │                │                             │               │    
                    ┌─┴─┐       ┌─┴─┐              │                             │               │ 
                    │ b │       │ c │              │                          ┌──┴───┐        ┌──┴───┐
                    └───┘       └───┘              │                          │ ∂(y) │        │ ∂(y) │
                                                   │                          │ ──── │        │ ──── │
                                                   │                          │ ∂(b) │        │ ∂(c) │
                                                   │                          └──────┘        └──────┘
```

To accomplish this there is one key observation that needs to be made. Comparing 
the two graphs we can see that each node in the input graph corresponds to a node 
in the backward pass. This means that each input & operation that we use to build 
the computational graph representing our forward pass, should have a corresponding 
value in the computational graph represnting our backward pass. 

## Reading Guide 
The files represented here make up the core of ember and just understanding these 
files can facilitate a great understanding of automatic differentiation. To make 
this as easy as possible, I'd recommend the below order for reading:

1. **node.h** - Contains the `Node` struct, representing a node in the computational 
graph. These are the generic functions that (usually) correspond to a forward operation 
and calculate the gradients of it's inputs with respect to the output.
2. **edge.h** - Contains the `Edge` struct, representing an edge in the computational 
graph. `Edge`s connect `Node` to their input's'. You can think of edges as the 
pathways along which gradients are passed. i.e. if we have a node `A` with an edge 
connecting it to a node `B` then you can assume that `A` passes a gradient to `B`. 

With just these two files you should be able to understand how computational graphs 
(like the ones shown above) are constructed, next are the files that contain the 
logic for using these computational graphs to perform autodiff. 

3. **engine.h** - The core of autodiff. The `Engine` class is a *very* small 
set of code that propagates gradients through the computational graph by executing 
each node and padding the computed gradients along it's edges to the next nodes.
4. **accumulator.h** - Accumulator is an interesting one. If you've really read 
all the other files in this folder, you'll likely have a few questions. The 
`Accumulator` class should hopefully answer those outstanding questions. This 
could in theory be read right after edge or node but it's not a priority.

## Usage 

User's of the library won't need to use any of these files or the components they
contstruct directly. Instead these components are triggered behind the scenes 
when operations on tensors are performed or when the tensors `backward` method 
is invoked.

On the other hand, for persons seeking to extend the library with additional 
functionality -- especially in the case of defining new operations -- the following 
details the usage of key components here.

```c++
// defining a new operation.
Tensor NewOp(Tensor& a, Tensor& b) {
    Tensor c = ...; // operation implemenation details;
    // creating the node corresponding to the backward pass for the operation 
    // that produced c. 
    c.gradient_fn = NewOpBackward(); 
    // defining the edges that connect the backward function's node to the input 
    // tensors' nodes thereby connecting it to the compgraph.
    c.add_next_edge(Edge(0, a.get_gradient_edge()));
    c.add_next_edge(Edge(1, a.get_gradient_edge()));
}

// defining the node representing the gradient calculation for this operation.
struct NewOpBackward: public Node {
    std::vector<Tensor> operator()(Tensor output) {
        // Provide backward operation implemenation ...
    } 
}
```

## Common Issues 
1. One thing that can get confusing is referring to the gradient value that each 
backward fn receives as an output grad since technically that value is an input 
to the backward function, it's probably best to provide an explanation for this.
