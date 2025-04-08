"""
See section 4.5: Support for XLNS Objects
Adds support for operations between LNSTensor
and the global precision xlns objects (xlns,
xlnsud, xlnsnp). We can't support xlns objects
in torch functions but can in dunder methods.
"""
import functools
import contextlib
import numpy as np
import xlns as xl
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

def lnstensor(data, from_lns=False):
    if isinstance(data, LNSTensor):
        input_data = data._lns
        from_lns = True
    elif isinstance(data, torch.Tensor):
        input_data = data.to(torch.float64)
    elif isinstance(data, np.ndarray):
        input_data = torch.from_numpy(data).to(torch.float64)
    # For xlns non-redundant, global precision scalars
    elif isinstance(data, (xl.xlns, xl.xlnsud)):
        input_data = torch.tensor(data.x << 1 + data.s, dtype=torch.float64)
        from_lns = True
    # For xlns non-redundant, global precision numpy-like arrays
    elif isinstance(data, xl.xlnsnp):
        input_data = torch.tensor(data.nd, dtype=torch.float64)
        from_lns = True
    # For scalars, lists, or tuples, etc
    else:
        input_data = torch.tensor(data, dtype=torch.float64)
    return LNSTensor(input_data, from_lns=from_lns)

class LNSTensor(object):

    def __init__(self, data, from_lns=False):

        if from_lns:
            self._lns = data
        # We use 'with torch.no_grad()' here because the conversion
        # from floating-point to LNS shouldn't be tracked by autograd.
        else:
            with torch.no_grad():
                log_data = (torch.log(torch.abs(data))/torch.log(base))
                quant_log_data = log_data.round().to(torch.int64)
                sign_bit = (data <= 0).to(torch.int64)
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
        return lnstensor(self._lns.grad, from_lns=True)

    def __repr__(self):
        return f"LNSTensor({self._lns.to(torch.int64).numpy()})"

    def __mul__(self, other):
        return torch.mul(self, lnstensor(other))

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
    out_lns = LNSMulFunction.apply(input._lns, other._lns)
    return lnstensor(out_lns, from_lns=True)

x = lnstensor([1, 2])
y = xl.xlnsnp([2, 1])
z = xl.xlns(2)

u = x * y
v = x * z
w = x * 2

print("Multiply x and y element wise:", u)
print(f"Multiply x by 2 (xlns and scalar):\n", v, w)

# Output:
# Multiply x and y element wise: LNSTensor([16777216 16777216])
# Multiply x by 2 (xlns and scalar):
# LNSTensor([16777216 33554432]) LNSTensor([16777216 33554432])