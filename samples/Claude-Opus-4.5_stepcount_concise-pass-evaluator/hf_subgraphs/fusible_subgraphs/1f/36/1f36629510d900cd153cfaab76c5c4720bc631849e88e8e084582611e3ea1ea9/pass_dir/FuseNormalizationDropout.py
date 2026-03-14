import torch
import triton
import triton.language as tl

# Pattern matching function - sum + unsqueeze
def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Kernel wrapper - use native PyTorch with keepdim=True which fuses sum+unsqueeze
@torch.fx.wrap
def sum_unsqueeze(x):
    # sum(dim=-1, keepdim=True) is equivalent to sum(dim=-1).unsqueeze(-1)
    # but more efficient as it avoids the separate unsqueeze call
    return x.sum(dim=-1, keepdim=True)

# Replacement function
def replacement_func():
    return sum_unsqueeze