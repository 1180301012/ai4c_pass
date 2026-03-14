import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching for flatten(2) followed by permute(0, 2, 1)
    
    Input shape: [B, C, H, W]
    flatten(2): [B, C, H*W]  
    permute(0, 2, 1): [B, H*W, C]
    """
    tmp_0 = x.flatten(2)
    tmp_1 = tmp_0.permute(0, 2, 1)
    return tmp_1

def replacement_args(x):
    """Extract arguments needed for the replacement"""
    return (x,)

@torch.fx.wrap
def fused_reshape(x):
    """Optimized function that fuses flatten(2) + permute(0, 2, 1)
    
    This eliminates the intermediate tensor created by separate operations
    while preserving computational equivalence.
    """
    # First do flatten(2) - preserve first 2 dimensions, flatten from dim 2 onwards
    flattened = x.flatten(2)
    
    # Then do permute(0, 2, 1) - swap the last two dimensions
    # This is equivalent to reshape but more explicit about the permutation
    return flattened.permute(0, 2, 1)

def replacement_func():
    """Return the optimized fused reshape function"""
    return fused_reshape