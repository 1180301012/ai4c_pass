import torch
import triton
import triton.language as tl
import numpy as np

def pattern(in_0):
    """Match flatten(2) followed by permute(0, 2, 1) pattern"""
    tmp_0 = in_0.flatten(2)
    tmp_1 = tmp_0.permute(0, 2, 1)
    return tmp_1

def replacement_args(in_0):
    """Extract the input tensor argument for the optimized kernel"""
    return (in_0,)

@torch.fx.wrap
def optimized_flatten_permute(in_0):
    """Optimized implementation that maintains exact equivalence"""
    # The original computation: flatten(2) then permute(0, 2, 1)
    # Convert [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
    #
    # We can optimize this by doing the operations in a more efficient sequence:
    # 1. First permute to get [B, H, W, C] 
    # 2. Then reshape to [B, H*W, C]
    # This avoids the intermediate tensor from flatten(2) and achieves the same result
    
    # Step 1: Transpose to [B, H, W, C] 
    tmp_0 = in_0.permute(0, 2, 3, 1)
    
    # Step 2: Reshape from [B, H, W, C] to [B, H*W, C]
    return tmp_0.reshape(in_0.shape[0], in_0.shape[2] * in_0.shape[3], in_0.shape[1])

def replacement_func():
    """Return the optimized function"""
    return optimized_flatten_permute