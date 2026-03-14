# This file contains the optimization pass for FuseFlattenTransposeToReshape
# The pass replaces flatten(2) + transpose(1,2) with permute + contiguous + view
# Configuration file reference: sorted_output_pass_rule_names.json
import torch
import triton
import triton.language as tl

# Pattern matching: flatten(2) followed by transpose(1,2)
# Input shape: (N, C, H, W) = (1, 768, 14, 14)
# flatten(2): (N, C, H*W) = (1, 768, 196)
# transpose(1,2): (N, H*W, C) = (1, 196, 768)
#
# OPTIMIZATION STRATEGY:
# Instead of a custom Triton kernel (which has overhead), we use PyTorch's
# built-in permute + reshape which are highly optimized and don't require
# kernel compilation overhead.

def pattern(conv_out):
    """
    Match flatten(2) then transpose(1,2) pattern.
    conv_out: (N, C, H, W) e.g., (1, 768, 14, 14)
    """
    # First: flatten(2) - flatten from dimension 2 onwards
    # Result: (N, C, H*W)
    flat = conv_out.flatten(2)
    
    # Second: transpose(1, 2) - swap C and H*W dimensions
    # Result: (N, H*W, C)
    transposed = flat.transpose(1, 2)
    
    return transposed


def replacement_args(conv_out):
    """Extract the argument needed for replacement."""
    return (conv_out,)


def replacement_func():
    """Return the optimized function."""
    return efficient_flatten_transpose


@torch.fx.wrap
def efficient_flatten_transpose(conv_out):
    """
    Optimized implementation using PyTorch's permute + contiguous + reshape.
    
    This avoids the overhead of:
    1. Creating intermediate tensor from flatten
    2. Creating another intermediate from transpose
    
    By using permute+reshape, we get better memory access patterns
    and let PyTorch's fusion optimizer handle the rest.
    """
    N, C, H, W = conv_out.shape
    
    # permute(0, 2, 3, 1) transforms (N, C, H, W) -> (N, H, W, C)
    # This requires a copy to make it contiguous
    # reshape(N, H*W, C) transforms (N, H, W, C) -> (N, H*W, C)
    return conv_out.permute(0, 2, 3, 1).contiguous().view(N, H * W, C)