import torch
import triton
import triton.language as tl

# Simple pattern - just match silu
def pattern(x):
    return torch.nn.functional.silu(x, inplace=True)

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def silu_wrapper(x):
    return torch.nn.functional.silu(x, inplace=True)

def replacement_func():
    return silu_wrapper