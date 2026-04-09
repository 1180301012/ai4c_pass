import torch

def pattern(x):
    """Pattern to match: two consecutive reshape operations that can be fused"""
    tmp_1 = x.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2

def replacement_args(x):
    return (x,)

def optimized_fused_reshape(x):
    """Optimized fused reshape operation - minimal overhead version"""
    # Direct reshape with pre-computed target for maximum efficiency
    # Eliminating runtime calculations for better performance
    return x.reshape(1, 248, 768)

def replacement_func():
    return optimized_fused_reshape