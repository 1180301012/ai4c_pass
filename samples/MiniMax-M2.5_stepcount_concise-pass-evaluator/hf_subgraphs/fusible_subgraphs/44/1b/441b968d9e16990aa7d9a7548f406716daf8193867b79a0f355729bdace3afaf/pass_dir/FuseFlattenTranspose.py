import torch
import triton
import triton.language as tl

# Pattern: flatten(2) -> transpose(1, 2)
# This can be simplified to a single reshape:
# in_4: [batch, 64, 128, 128]
# flatten(2): [batch, 64, 16384]
# transpose(1,2): [batch, 16384, 64]
# Equivalent to: reshape(batch, 16384, 64) = view(batch, -1, 64)

def pattern(in_4):
    """
    Match pattern: flatten(2) -> transpose(1, 2)
    This pattern appears in the model:
    tmp_7 = in_4.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    """
    tmp_7 = in_4.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8

def replacement_args(in_4):
    # Extract the input tensor
    return (in_4,)

def fused_flatten_transpose(x):
    """
    Optimized implementation: single reshape operation
    Original: flatten(2).transpose(1, 2)
    Input: [batch, 64, 128, 128]
    flatten(2): [batch, 64, 16384]
    transpose(1,2): [batch, 16384, 64]
    Equivalent to: reshape(batch, 16384, 64)
    """
    B, C, H, W = x.shape
    # Reshape directly to (batch, H*W, C)
    # flatten(2) + transpose(1,2) = reshape(batch, H*W, C)
    return x.reshape(B, H * W, C)

def replacement_func():
    return fused_flatten_transpose