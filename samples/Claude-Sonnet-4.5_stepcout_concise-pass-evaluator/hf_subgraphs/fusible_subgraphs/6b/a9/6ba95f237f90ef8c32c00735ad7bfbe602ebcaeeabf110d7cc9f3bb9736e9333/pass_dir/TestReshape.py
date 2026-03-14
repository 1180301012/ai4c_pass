import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern to match: reshape only
    """
    tmp_3 = input_tensor.reshape(1, 128, 4, -1)
    return (tmp_3,)

def replacement_args(input_tensor):
    return (input_tensor,)

@torch.fx.wrap
def optimized_reshape(input_tensor):
    """
    Just a passthrough for now to test matching
    """
    return input_tensor.reshape(1, 128, 4, -1)

def replacement_func():
    return optimized_reshape