import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = x
    tmp_1 = tmp_0.exp()
    return tmp_1

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_scalar_exp(x):
    # For scalar, just use native torch.exp which is already optimized
    return torch.exp(x)

def replacement_func():
    return optimized_scalar_exp