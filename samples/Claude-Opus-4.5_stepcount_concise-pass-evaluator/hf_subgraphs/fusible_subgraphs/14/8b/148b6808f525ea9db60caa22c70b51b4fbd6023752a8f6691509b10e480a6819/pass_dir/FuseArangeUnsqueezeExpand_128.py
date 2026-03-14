import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function - matches unsqueeze + expand on the tensor
def pattern(x):
    tmp_1 = x.unsqueeze(0)
    tmp_2 = tmp_1.expand(1, -1)
    return tmp_2

# Argument extraction function
def replacement_args(x):
    return (x,)

# Use view with explicit dimensions - may be faster than unsqueeze
@torch.fx.wrap  
def fast_unsqueeze_expand(x):
    return x.view(1, -1)

# Replacement function returns the wrapper function reference
def replacement_func():
    return fast_unsqueeze_expand