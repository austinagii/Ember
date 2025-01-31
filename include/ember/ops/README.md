# Ops: Operations on Tensors

## Overview

The `ops` folder contains files that define the core operations that can be 
performed on tensors. Operations are defined as binary operations that take two 
tensors as input and return a new tensor as output. Each operation is defined 
in a separate file, each of which contains both the forward pass and the 
corresponding gradient computation / backward pass.

## Reading Guide

The following order is recommended for reading the code:
1. `add.h`: The addition operation is the easiest to understand and reading 
this first should give you a good understanding of how each operation is 
defined.
2. `mul.h`: The multiplication operation is next. With a solid understanding 
of the basics from addition, this should now show how `TensorSnapshot`s are used 
for backward passes that need them.
3. `div.h`: The division operation is next. This has a slightly more complex 
backward pass than the others but mostly due the the actual maths for calculating 
the derivative. Otherwise, it's a simple operation, marginally more complex than 
multiplication.
4. `sub.h`: The subtraction operation is fairly basic, it's addition with a twist. 
Lower on the list since it doesn't provide much more value than the addition.

## Usage 

As mentioned before, all operations are binary operations. As such, all of them
can be used as part of a binary expression: `Tensor <op> Tensor`.  

Examples:
```c++
Tensor a {2.0};
Tensor b {1.0};

a + b; // addition
a - b; // subtraction
a * b; // multiplication
a / b; // division
```

## Testing

Each operation has a suite of tests (found at `${PROJECT_ROOT}/tests/ember/ops`)
that aims to provide comprehensive coverage of the various cirumstances under 
which these operations can be used, such as:
- Scalar operations
- Multi-dimensional operations
- Broadcasting scenarios
- Single & nested / composite operations
- Edge cases
- Gradient computation verification for all the above

## Common Issues
1. **Broadcasting Rules**: When operating on tensors of different shapes, broadcasting follows these rules:
   - Smaller tensors are broadcast to match larger ones
   - Shapes must be compatible for broadcasting
   - Gradients are properly accumulated across broadcast dimensions

2. **Division by Zero**: Division operations include checks to prevent division by zero:
