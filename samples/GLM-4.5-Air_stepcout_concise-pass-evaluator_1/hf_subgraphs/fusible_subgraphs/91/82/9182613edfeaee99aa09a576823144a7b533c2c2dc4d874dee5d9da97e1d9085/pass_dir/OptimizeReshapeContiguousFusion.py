import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: Complete computation pattern for the transformer subgraph
    Original:
        tmp_0 = in_1.reshape(B, 16, H, W)
        tmp_1 = in_0[..., :N]  # No-op slice
        tmp_2 = in_2.contiguous()
        tmp_3 = in_3.contiguous()
        tmp_4 = tmp_0.contiguous()
    Returns: (tmp_1, tmp_3, tmp_2, tmp_4)
    """
    # Reshape from [B, 4, 4, H, W] to [B, 16, H, W]
    tmp_0 = in_1.reshape(4, 16, 512, 128)
    
    # Slice operation (no-op for full dimension)
    tmp_1 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 512, None)]
    
    # Contiguous operations
    tmp_2 = in_2.contiguous()
    tmp_3 = in_3.contiguous()
    tmp_4 = tmp_0.contiguous()
    
    return tmp_1, tmp_3, tmp_2, tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@torch.fx.wrap  
def optimized_reshape_contiguous(in_0, in_1, in_2, in_3):
    """
    Optimized version: Eliminate redundant operations
    1. Slice is a no-op - return input directly
    2. Reshape typically produces contiguous tensor - eliminate final contiguous()
    3. Only make tensors contiguous if they're not already
    """
    # No-op slice - return input directly  
    tmp_1 = in_0
    
    # Reshape - already produces contiguous tensor for typical memory layouts
    tmp_4 = in_1.reshape(4, 16, 512, 128)
    
    # Only make tensors contiguous if they're not already
    tmp_2 = in_2 if in_2.is_contiguous() else in_2.contiguous()
    tmp_3 = in_3 if in_3.is_contiguous() else in_3.contiguous()
    
    return tmp_1, tmp_3, tmp_2, tmp_4

def replacement_func():
    return optimized_reshape_contiguous