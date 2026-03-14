"""
FuseFlattenTranspose Optimization Pass

This pass optimizes the flatten + transpose pattern commonly found in vision model
decode heads by using view + permute instead of flatten + transpose.

Pattern: flatten(2) + transpose(1, 2) -> [B, C, H, W] -> [B, H*W, C]
Replacement: view + permute -> [B, C, H*W] -> [B, H*W, C]
"""

import torch
import triton
import triton.language as tl


def pattern(in_2):
    """Match: flatten(2) followed by transpose(1, 2)"""
    tmp_6 = in_2.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7


def replacement_args(in_2):
    """Extract input tensor argument"""
    return (in_2,)


def replacement_func():
    """Return optimized replacement using view + permute"""
    def optimized_reshape_to_sequence(x):
        B, C, H, W = x.shape
        x_viewed = x.view(B, C, H * W)
        return x_viewed.permute(0, 2, 1)
    
    return optimized_reshape_to_sequence