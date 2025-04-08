"""
See section 4.4: Alternative Operator Implementations
This adds support for multiple implementations of the
same operator. We can set a default implementation and
override it for specific calls with a context manager.
"""
import functools
import contextlib
import numpy as np
import torch

f = 23
base = torch.tensor(2.0**(2**(-f)), dtype=torch.float64)

HANDLED_FUNCTIONS = {}
DEFAULT_IMPLEMENTATIONS = {}

# This is a decorator that will do the work under the hood
# of registering a function with the global dispatch table.
def implements(torch_function, key=None, default=False):
    """Register a torch function override for LNSTensor."""
    def decorator(func):
        function_key = key or func.__name__
        functools.update_wrapper(func, torch_function)
        if torch_function not in HANDLED_FUNCTIONS:
            HANDLED_FUNCTIONS[torch_function] = {}
        HANDLED_FUNCTIONS[torch_function][function_key] = func
        if default:
            DEFAULT_IMPLEMENTATIONS[torch_function] = key
        return func
    return decorator

def set_default(torch_function, impl_key):
    if torch_function not in HANDLED_FUNCTIONS:
        raise ValueError("No implementations registered for the given torch function.")
    if impl_key not in HANDLED_FUNCTIONS[torch_function]:
        raise ValueError(f"Implementation '{impl_key}' is not registered for {torch_function}.")
    DEFAULT_IMPLEMENTATIONS[torch_function] = impl_key

@contextlib.contextmanager
def override_impl(torch_function, impl_key):
    """Override the default implementation for a specific torch function."""
    original_default = DEFAULT_IMPLEMENTATIONS.get(torch_function)
    set_default(torch_function, impl_key)
    try:
        yield
    finally:
        set_default(torch_function, original_default)

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
        chosen_impl = DEFAULT_IMPLEMENTATIONS.get(func)
        if chosen_impl is None:
            raise RuntimeError(f"No default implementation has been set for {func}.")
        if chosen_impl not in HANDLED_FUNCTIONS[func]:
            raise ValueError(f"Implementation key '{chosen_impl}' is not registered for {func}.")
        return HANDLED_FUNCTIONS[func][chosen_impl](*args, **kwargs)

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

@implements(torch.mul, key="default", default=True)
def multiply(input, other):
    print("Using default implementation")
    out_lns = LNSMulFunction.apply(input._lns, other._lns)
    return LNSTensor(out_lns, from_lns=True)

@implements(torch.mul, key="alternative_1")
def multiply_alternative_1(input, other):
    print("Using alternative implementation")
    out_lns = LNSMulFunction.apply(input._lns, other._lns)
    return LNSTensor(out_lns, from_lns=True)

set_default(torch.mul, "alternative_1") # switches the default to some alternative implementation

x = LNSTensor(1)
y = LNSTensor(2)

z = torch.mul(x, y) # uses alternative_1
with override_impl(torch.mul, "default"):
    w = torch.mul(x, y) # uses the original default implementation

# Output:
# Using alternative implementation
# Using default implementation