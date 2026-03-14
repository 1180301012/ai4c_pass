import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: match contiguous operation
    """
    y = x.contiguous()
    return y

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_contiguous(x):
    """
    Skip contiguous() if tensor is already contiguous.
    This avoids PyTorch's internal overhead.
    """
    # Fast path: if already contiguous, return as-is
    if x.is_contiguous():
        return x
    
    # Otherwise, use PyTorch's optimized contiguous()
    # Don't use custom Triton kernel as it's slower
    return x.contiguous()

def replacement_func():
    return optimized_contiguous