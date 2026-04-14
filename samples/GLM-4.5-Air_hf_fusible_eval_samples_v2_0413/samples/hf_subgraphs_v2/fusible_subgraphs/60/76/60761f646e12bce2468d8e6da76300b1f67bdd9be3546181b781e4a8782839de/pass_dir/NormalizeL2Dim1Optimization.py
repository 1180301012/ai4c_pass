import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple test pattern to see if basic matching works"""
    # Try something extremely simple
    return torch.nn.functional.normalize(x, p=2, dim=1)

def replacement_args(x):
    return (x,)

def replacement_func():
    def simple_normalize(x):
        return torch.nn.functional.normalize(x, p=2, dim=1)
    return simple_normalize