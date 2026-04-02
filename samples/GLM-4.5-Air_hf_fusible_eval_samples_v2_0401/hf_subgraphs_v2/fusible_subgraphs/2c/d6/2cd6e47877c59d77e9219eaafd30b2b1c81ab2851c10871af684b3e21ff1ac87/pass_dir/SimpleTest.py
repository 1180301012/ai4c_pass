import torch

def pattern(in_3, in_1):
    # Simple multiplication pattern
    return in_3 * in_1

def replacement_args(in_3, in_1):
    return (in_3, in_1)

def simple_mult_native(x, y):
    """Native PyTorch multiplication for perfect correctness"""
    return x * y

@torch.fx.wrap
def simple_mult_optimized(x, y):
    """Use native PyTorch multiplication to ensure correctness"""
    return simple_mult_native(x, y)

def replacement_func():
    return simple_mult_optimized