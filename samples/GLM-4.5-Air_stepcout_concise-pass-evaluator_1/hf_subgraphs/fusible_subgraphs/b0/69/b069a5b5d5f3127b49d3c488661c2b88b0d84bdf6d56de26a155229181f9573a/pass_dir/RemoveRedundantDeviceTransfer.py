import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function - matches the redundant device transfer
def pattern(in_1):
    return in_1.to(device(type='cuda', index=0))

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Identity kernel - just return the input since no conversion needed
@torch.fx.wrap
def identity_wrapper(in_1):
    return in_1

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return identity_wrapper