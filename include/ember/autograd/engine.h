#ifndef ENGINE_H
#define ENGINE_H

#include <ember/tensor.h>
#include <ember/autograd/node.h>

namespace ember::autograd {
    class Engine {
    public:
        static void evaluate_fn(Node* func, ember::Tensor output_grad);
    };

} // namespace ember::autograd

#endif // !ENGINE_H
