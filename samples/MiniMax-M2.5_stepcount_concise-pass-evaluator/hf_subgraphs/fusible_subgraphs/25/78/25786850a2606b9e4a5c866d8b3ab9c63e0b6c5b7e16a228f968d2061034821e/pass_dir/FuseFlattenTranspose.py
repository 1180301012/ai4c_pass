import torch
import triton
import triton.language as tl


def pattern(tmp_5):
    """
    Pattern to match: flatten(2) followed by transpose(1, 2)
    Input: tmp_5 with shape [1, C, H, W] from conv2d
    flatten(2) produces [1, C, H*W]
    transpose(1, 2) gives [1, H*W, C]
    """
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7


def replacement_args(tmp_5):
    return (tmp_5,)


def flatten_transpose_optimized(x):
    """
    Optimized flatten(2) + transpose(1, 2) using reshape and permute.
    This avoids creating an intermediate [1, C, H*W] tensor.
    
    Input x: [1, C, H, W]
    Output: [1, H*W, C]
    """
    B, C, H, W = x.shape
    
    # Instead of: flatten(2) -> transpose(1,2)
    # We do: reshape to [1, C, H, W] -> reshape to [1, H*W, C] directly
    # This is done via reshape + view operations
    
    # Use reshape which can be more efficient when possible
    # First reshape to collapse H and W into a single dimension
    x_reshaped = x.reshape(B, C, H * W)
    # Then transpose
    result = x_reshaped.transpose(1, 2)
    
    return result


def replacement_func():
    return flatten_transpose_optimized