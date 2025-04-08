"""
See section 4.2: Problems with Integer Tensor Differentiation
This demonstrates how PyTorch doesn't support differentiation
for integer tensors.
"""
import torch

x = torch.tensor(1, requires_grad=True)
y = torch.tensor(2, requires_grad=True)

z = x + y
z.backward()

# Output:
# RuntimeError: Only Tensors of floating point and complex dtype can require gradients