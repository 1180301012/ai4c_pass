import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the pattern: x -> unsqueeze(0) -> expand(1, -1)
    
    # Execute the operations we want to optimize
    tmp_1 = x.unsqueeze(0)
    tmp_2 = tmp_1.expand(1, -1)
    
    return tmp_2

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_unsqueeze(x):
    # The optimization: unsqueeze(0) followed by expand(1, -1) is redundant
    # We can just use unsqueeze(0) directly since expand(1, -1) is a no-op
    # when the first dimension is already 1
    #
    # For small tensors like this (128 elements), using PyTorch's native
    # unsqueeze operation is more efficient than a custom Triton kernel
    # because the kernel launch overhead would outweigh the benefits
    
    # Create the unsqueezed result directly - this replaces the entire sequence
    # x.unsqueeze(0).expand(1, -1) with just x.unsqueeze(0)
    return x.unsqueeze(0)

def replacement_func():
    return optimized_unsqueeze