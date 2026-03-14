import torch
import triton
import triton.language as tl

# Simple add pattern
def pattern(a, b):
    return a + b

def replacement_args(a, b):
    return (a, b)

@torch.fx.wrap  
def optimized_add(a, b):
    # Use Python + operator which handles broadcasting
    return a + b

def replacement_func():
    return optimized_add