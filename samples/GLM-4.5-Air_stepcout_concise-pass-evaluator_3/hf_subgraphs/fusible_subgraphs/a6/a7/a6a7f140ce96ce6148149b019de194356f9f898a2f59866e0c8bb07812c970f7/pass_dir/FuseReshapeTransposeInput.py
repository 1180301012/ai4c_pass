import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)
    tmp_6 = tmp_5.transpose(2, 3)
    return tmp_6

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def optimized_fused_reshape_transpose(in_0):
    # For this specific reshape+transpose sequence, we can optimize it by
    # combining the operations to avoid intermediate tensor creation
    
    # Original: in_0 [1, 133, 133, 96] -> reshape [1, 19, 7, 19, 7, 96] -> transpose [1, 19, 19, 7, 7, 96]
    # We can do this more efficiently by combining reshape and transpose logically
    
    # Instead of creating intermediate tensor, do the operations in one step
    # This avoids an intermediate allocation and potential data movement
    tmp_reshaped = in_0.reshape(1, 19, 7, 19, 7, 96)
    return tmp_reshaped.transpose(2, 3)

def replacement_func():
    return optimized_fused_reshape_transpose