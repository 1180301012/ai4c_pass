import torch

def pattern(in_tensor):
    # RoPE slice pattern: extract odd/even indices, negate, and stack
    tmp_2 = in_tensor[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_tensor[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    return tmp_5

def replacement_args(in_tensor):
    return (in_tensor,)

def rope_slice_native(x):
    """Native implementation of RoPE slice operations for perfect correctness"""
    # Extract odd indices (starting from index 1, step 2)
    odd_slice = x[(Ellipsis, slice(1, None, 2))]
    # Negate the odd slice
    negated_odd = -odd_slice
    # Extract even indices
    even_slice = x[(Ellipsis, slice(None, None, 2))]
    # Stack negated odd and even slices along last dimension
    result = torch.stack([negated_odd, even_slice], -1)
    return result

@torch.fx.wrap
def rope_slice_optimized(x):
    """Use native implementation to ensure correctness"""
    return rope_slice_native(x)

def replacement_func():
    return rope_slice_optimized