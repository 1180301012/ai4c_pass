import torch
import triton
import triton.language as tl

# Match the scalar multiplication operation
def pattern(in_0, in_1):
    """Pattern to match scalar multiplication"""
    result = in_0 * in_1
    return result

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap  
def optimized_mul(in_0, in_1):
    """Optimized scalar multiplication - direct computation"""
    return in_0 * in_1

def replacement_func():
    return optimized_mul