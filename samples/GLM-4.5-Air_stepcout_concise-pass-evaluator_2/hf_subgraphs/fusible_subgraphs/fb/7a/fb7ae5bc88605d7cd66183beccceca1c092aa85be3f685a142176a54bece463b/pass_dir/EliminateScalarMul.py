import torch

def pattern(in_0):
    tmp_0 = in_0 * 1.0
    return tmp_0

@torch.fx.wrap
def eliminate_scalar_mul(in_0):
    """Eliminate scalar multiplication by 1.0"""
    return in_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return eliminate_scalar_mul