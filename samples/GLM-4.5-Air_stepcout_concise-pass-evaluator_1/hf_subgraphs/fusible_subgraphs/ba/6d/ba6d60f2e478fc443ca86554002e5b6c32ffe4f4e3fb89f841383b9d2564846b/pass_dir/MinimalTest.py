import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Minimal test pattern: just a simple multiply"""
    return x * y

def replacement_args(x, y):
    return (x, y)

@torch.fx.wrap
def minimal_multiply(x, y):
    """Just a wrapper around torch multiply"""
    return x * y

def replacement_func():
    return minimal_multiply