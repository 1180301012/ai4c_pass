import torch
import triton
import triton.language as tl

# Pattern matching function - just flatten
def pattern(in_0):
    tmp_0 = in_0.flatten(1, -1)
    return tmp_0

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Simple flatten implementation - use view for efficiency (no data copy)
@torch.fx.wrap
def efficient_flatten(in_0):
    # Get input shape info
    batch_size = in_0.shape[0]
    # Calculate flattened size (everything from dim 1 to end)
    flat_size = 1
    for i in range(1, len(in_0.shape)):
        flat_size *= in_0.shape[i]
    
    # Use view for zero-copy flatten (requires contiguous input)
    if in_0.is_contiguous():
        return in_0.view(batch_size, flat_size)
    else:
        return in_0.contiguous().view(batch_size, flat_size)

def replacement_func():
    return efficient_flatten