import torch
import triton
import triton.language as tl

def pattern(x):
    result = x.mean(dim=-2, keepdim=True)
    return result

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def mean_dim2_triton(x):
    return x.mean(dim=-2, keepdim=True)

def replacement_func():
    return mean_dim2_triton