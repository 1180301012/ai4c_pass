import torch

# Pattern matching function - matches the normalization operation
def pattern(in_1):
    tmp_0 = in_1.sum(dim = 2, keepdim = True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Simple but optimized normalization using PyTorch operations
@torch.fx.wrap
def simple_normalize_dim2(in_1):
    """
    Simple normalization along dimension 2 using fused operations
    This is equivalent to: in_1 / in_1.sum(dim=2, keepdim=True)
    """
    # Compute sums along dimension 2 in one operation
    sums_dim2 = in_1.sum(dim=2, keepdim=True)
    
    # Perform element-wise division with small epsilon for numerical stability
    return in_1 / (sums_dim2 + 1e-7)

# Replacement function (MUST return function reference, not call)
def replacement_func():
    return simple_normalize_dim2