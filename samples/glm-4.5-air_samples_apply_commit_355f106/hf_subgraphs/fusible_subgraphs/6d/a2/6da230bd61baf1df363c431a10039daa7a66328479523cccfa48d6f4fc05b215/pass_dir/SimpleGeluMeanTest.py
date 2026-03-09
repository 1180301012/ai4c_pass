import torch
import triton
import triton.language as tl

# Simple pattern matching function
def pattern(x):
    return torch.nn.functional.gelu(x), x.mean((2, 3), keepdim=True)

# Extract arguments
def replacement_args(x):
    return (x,)

# Simple wrapper for testing
@torch.fx.wrap
def simple_gelu_mean(x):
    return torch.nn.functional.gelu(x), x.mean((2, 3), keepdim=True)

# Replacement function
def replacement_func():
    return simple_gelu_mean