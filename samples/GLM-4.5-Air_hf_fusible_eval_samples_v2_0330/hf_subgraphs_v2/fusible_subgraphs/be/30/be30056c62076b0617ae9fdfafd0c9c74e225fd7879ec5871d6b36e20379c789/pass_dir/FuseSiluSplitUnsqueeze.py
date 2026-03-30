import torch
import triton
import triton.language as tl

def pattern(tensor):
    """Simple pattern - just a single tensor operation"""
    return tensor + 1

def replacement_args(x):
    return (x,)

@torch.fx.wrap  
def simple_tensor_operation(x):
    """Simple tensor operation - should match the pattern"""
    return x + 1

def replacement_func():
    return simple_tensor_operation