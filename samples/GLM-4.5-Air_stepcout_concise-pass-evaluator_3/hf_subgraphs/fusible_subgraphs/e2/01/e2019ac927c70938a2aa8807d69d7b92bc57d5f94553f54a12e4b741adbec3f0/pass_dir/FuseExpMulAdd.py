import torch


# Pattern matching function - matches the computation: exp(in_1) * in_2 + in_0
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    return tmp_2


# Extract arguments needed for replacement
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Simple wrapper that uses PyTorch's native operations efficiently
@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Optimized computation using PyTorch's fused operations.
    """
    # Get scalar values efficiently
    scale = in_1.exp()
    
    # Use fused multiply-add pattern which PyTorch can optimize
    result = in_2 * scale + in_0
    
    return result


def replacement_func():
    return fused_kernel_wrapper