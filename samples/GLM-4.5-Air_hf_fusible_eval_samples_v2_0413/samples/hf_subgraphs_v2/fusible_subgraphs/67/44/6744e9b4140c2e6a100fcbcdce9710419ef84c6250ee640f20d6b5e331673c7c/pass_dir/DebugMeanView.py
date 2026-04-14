import torch
import triton
import triton.language as tl

# Pattern matching function for mean + view - trying to match exactly what we see in the models
def pattern(tmp_0):
    """Match the exact sequence from the model: tmp_1 = tmp_0.mean((2, 3)); tmp_4 = tmp_1.view(1, 1, -1)"""
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_1, tmp_4

# Argument extraction function
def replacement_args(tmp_0):
    return (tmp_0,)

# Simple test function to verify pattern works
@torch.fx.wrap
def debug_mean_view(tmp_0):
    """Debug version - just return what the original would return"""
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_1, tmp_4

# Replacement function (must return a callable function)
def replacement_func():
    return debug_mean_view