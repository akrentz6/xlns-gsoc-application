"""
See section 4.3: Overloading Operators
This demonstrates how we can overload PyTorch's
operators like torch.mul to work with LNSTensor
even though it isn't a subclass of torch.Tensor.
We use a global dispatch table as in the PyTorch
docs to register our custom functions.
(https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type)
"""
import functools
import numpy as np
import torch

f = 23
base = torch.tensor(2.0**(2**(-f)), dtype=torch.float64)

HANDLED_FUNCTIONS = {}

# This is a decorator that will do the work under the hood
# of registering a function with the global dispatch table.
def implements(torch_function):
    """Register a torch function override for LNSTensor."""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

class LNSTensor(object):

    def __init__(self, data, from_lns=False):

        # Convert data to a torch.Tensor of the correct type
        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                input_data = torch.from_numpy(data).to(torch.float64)
            else:
                # For scalars, lists, or tuples
                input_data = torch.tensor(data, dtype=torch.float64)
        else:
            input_data = data.to(torch.float64)

        if from_lns:
            self._lns = input_data
        # We use 'with torch.no_grad()' here because the conversion
        # from floating-point to LNS shouldn't be tracked by autograd.
        else:
            with torch.no_grad():
                log_data = (torch.log(input_data)/torch.log(base))
                quant_log_data = log_data.round().to(torch.int64)
                sign_bit = (input_data <= 0).to(torch.int64)
                self._lns = torch.bitwise_left_shift(quant_log_data, 1)
                self._lns += sign_bit
                self._lns = self._lns.to(torch.float64)

        self._lns.requires_grad_(True)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Check if we've registered a function for this operator and call it"""
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(issubclass(t, LNSTensor) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if gradient is None:
            gradient = torch.zeros_like(self._lns)
        return self._lns.backward(gradient, retain_graph, create_graph, inputs)

    @property
    def grad(self):
        if self._lns.grad is None:
            return None
        return LNSTensor(self._lns.grad, from_lns=True)

    def __repr__(self):
        return f"LNSTensor({self._lns.to(torch.int64).numpy()})"

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

@implements(torch.mul)
def multiply(input, other):
    out_lns = LNSMulFunction.apply(input._lns, other._lns)
    return LNSTensor(out_lns, from_lns=True)

x = LNSTensor(1) # Internal representation is LNSTensor(0)
y = LNSTensor(2) # Internal representation is LNSTensor(16777216)

z = torch.mul(x, y)
print("x * y =", z) # Internal representation is 1*2=2: LNSTensor(16777216)

z.backward()
print("Derivative w.r.t x:", x.grad)
print("Derivative w.r.t y:", y.grad)

# Output:
# x * y = LNSTensor(16777216)
# Derivative w.r.t x: LNSTensor(16777216)
# Derivative w.r.t y: LNSTensor(0)