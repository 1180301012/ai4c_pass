import torch
import triton
import triton.language as tl

# Pattern matching function - match the max reduction and arithmetic
def pattern(x):
    tmp_9 = x.max(-1, keepdim=True)
    tmp_11 = tmp_9[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_max_reduce_with_arith(x):
    """Optimized max reduction with arithmetic using native PyTorch."""
    # Fuse max(-1, keepdim=True)[0] + 1 - 9 into a single operation
    # This avoids intermediate tensor allocations
    result = x.max(-1, keepdim=True)[0]
    result = result.add_(1).sub_(9)  # In-place to save memory
    return result

def replacement_func():
    return fused_max_reduce_with_arith