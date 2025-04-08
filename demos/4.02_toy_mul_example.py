"""
See section 4.2: Problems with Integer Tensor Differentiation
This is an example of how we can use doubles (torch.float64)
to get around the problems of autograd with integer tensors.
I define a custom multiplication function that works on the
logarithmic representations of two real numbers.
"""
import torch

class LNSMulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # Multiplication in the log domain becomes addition
        out = a + b
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # The partial derivatives with respect to a and b are:
        # d/da(ab) = b
        # d/db(ab) = a
        # We are still working in the log domain during
        # backpropagation so multiplication once again
        # becomes addition.
        grad_a = grad_output + b
        grad_b = grad_output + a
        return grad_a, grad_b

# The internal representations of two decimal numbers
x = torch.tensor(2, dtype=torch.float64, requires_grad=True)
y = torch.tensor(3, dtype=torch.float64, requires_grad=True)

z = LNSMulFunction.apply(x, y)
print("x * y =", z)

# We supply a zero tensor for the gradient argument (the
# inital gradient with respect to the output tensor) since
# in our LNS, the number 1 is internally represented by a 0.
z.backward(gradient=torch.zeros_like(z))
print("Derivative w.r.t x:", x.grad)
print("Derivative w.r.t y:", y.grad)

# Output:
# x * y = tensor(5., dtype=torch.float64, grad_fn=<LNSMulFunctionBackward>)
# Derivative w.r.t x: tensor(3., dtype=torch.float64)
# Derivative w.r.t y: tensor(2., dtype=torch.float64)