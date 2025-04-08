"""
See section 4.8: Building Models
Gives a demo of how we would create custom
layers and optimizers that work with LNSTensor.
Note that we would need to implement subtraction
for TestOptimizer to work, but this is excluded
for length and the implementation for addition
is addressed in the coding challenges.
"""
import functools
import contextlib
import numpy as np
import xlns as xl
import torch

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

def lnstensor(data, from_lns=False, f=None, b=None):

    if f is not None and b is None:
        base = 2.0**(2**(-f))
    elif b is not None:
        base = b
    else:
        base = xl.xlnsB

    if isinstance(data, LNSTensor):
        input_data = data._lns
        from_lns = True
        if not base == data.base:
            pass # convert data representation from data.base to base

    elif isinstance(data, torch.Tensor):
        input_data = data.to(torch.float64)

    elif isinstance(data, np.ndarray):
        input_data = torch.from_numpy(data).to(torch.float64)

    # For xlns (non-redundant) scalars
    elif isinstance(data, (xl.xlns, xl.xlnsud, xl.xlnsv, xl.xlnsb)):
        if isinstance(data, (xl.xlns, xl.xlnsud)) and not base == xl.xlnsB:
            log_part = data.x * np.log(xl.xlnsB) / np.log(base)
        elif isinstance(data, (xl.xlnsb, xl.xlnsv)) and not base == data.B:
            log_part = data.x * np.log(data.B) / np.log(base)
        else:
            log_part = data.x
        input_data = torch.tensor(int(round(log_part)) << 1 + data.s, dtype=torch.float64)
        from_lns = True

    # For xlns (non-redundant) numpy-like arrays
    elif isinstance(data, (xl.xlnsnp, xl.xlnsnpv, xl.xlnsnpb)):
        data_x = data.nd >> 1
        data_s = data.nd & 1
        if isinstance(data, (xl.xlnsnp, xl.xlnsnpv)) and not base == xl.xlnsB:
            log_part = data_x * np.log(xl.xlnsB) / np.log(base)
        elif isinstance(data, xl.xlnsnpb) and not base == data.B:
            log_part = data_x * np.log(data.B) / np.log(base)
        else:
            log_part = data_x
        input_data = torch.tensor(np.int64(np.round(log_part)) << 1 + data_s, dtype=torch.float64)
        from_lns = True

    # For scalars, lists, or tuples, etc
    else:
        input_data = torch.tensor(data, dtype=torch.float64)

    base = torch.tensor(base, dtype=torch.float64)    
    return LNSTensor(input_data, base, from_lns=from_lns)

class LNSTensor(object):

    def __init__(self, data, base, from_lns=False):

        self.base = base

        if from_lns:
            self._lns = data
        # We use 'with torch.no_grad()' here because the conversion
        # from floating-point to LNS shouldn't be tracked by autograd.
        else:
            with torch.no_grad():
                log_data = (torch.log(torch.abs(data))/torch.log(self.base))
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
    other_aligned = lnstensor(other, b=input.base.item())
    out_lns = LNSMulFunction.apply(input._lns, other_aligned._lns)
    return LNSTensor(out_lns, from_lns=True, base=input.base)

class TestLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # generate random weights
        self.weights = lnstensor(torch.randn())

        # Wrap the internal representation in a Parameter
        self.weights._lns = torch.nn.Parameter(self.weights._lns)
        # Store a reference to the parameter so it's visible
        self.weights_param = self.weights._lns

class TestOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super(TestOptimizer, self).__init__(params, defaults)

    # For this toy example, ignore the closure parameter
    def step(self, closure=None):

        # Iterate over each parameter group
        for group in self.param_groups:

            lr = group["lr"]
            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.data
                with torch.no_grad():

                    # Wrap variables in the log-domain in LNSTensors
                    temp_data = lnstensor(p.data, from_lns=True)
                    temp_grad = lnstensor(grad, from_lns=True)

                    # Now uses LNSTensor operations
                    temp_data = temp_data - temp_grad * lr
                    p.data = temp_data._lns.data