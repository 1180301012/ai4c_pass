import torch
import triton
import triton.language as tl

def pattern(in_1):
    """Simple pattern to test detach operation matching"""
    tmp_1 = in_1.detach()
    return tmp_1

def replacement_args(in_1):
    """Extract arguments for the replacement kernel"""
    return (in_1,)

@torch.fx.wrap
def optimized_detach(in_1):
    """Simple optimized detach function"""
    # For now, just use the built-in detach to test pattern matching
    return in_1.detach()

def replacement_func():
    """Return the optimized function"""
    return optimized_detach