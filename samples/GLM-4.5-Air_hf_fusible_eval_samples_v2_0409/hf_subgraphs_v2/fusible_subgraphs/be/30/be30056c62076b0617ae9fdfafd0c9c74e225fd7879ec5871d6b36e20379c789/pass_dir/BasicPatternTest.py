import torch
import triton
import triton.language as tl

# Simple pattern that just matches tensor indexing
def pattern(x):
    return x[0]

# Extract arguments
def replacement_args(x):
    return (x,)

# Simple replacement that uses allowed APIs
@torch.fx.wrap
def simple_index(x):
    return x[0]

def replacement_func():
    return simple_index