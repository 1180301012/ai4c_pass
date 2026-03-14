import torch

def pattern(in_0, in_1, in_2):
    """Simple pattern: just multiplication and addition"""
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement"""
    return (in_0, in_1, in_2)

def replacement_func():
    """Return a simple optimized function"""
    def simple_fused_add_mul(x, y, z):
        """Simple fused multiplication and addition"""
        return (z * y) + x
    
    return simple_fused_add_mul