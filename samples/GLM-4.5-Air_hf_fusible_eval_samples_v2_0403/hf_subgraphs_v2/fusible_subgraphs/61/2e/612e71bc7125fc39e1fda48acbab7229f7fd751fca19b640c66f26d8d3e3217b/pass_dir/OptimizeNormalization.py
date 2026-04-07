import torch
from torch import device

def pattern(in_1):
    """
    Pattern matching for L2 normalization: norm(p=2, dim=-1, keepdim=True) followed by division
    """
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    """
    Extract input tensor for normalization
    """
    return (in_1,)

@torch.fx.wrap
def optimized_l2_norm(x):
    """
    Optimized L2 normalization using PyTorch's native operations
    For small tensors, native PyTorch is more efficient than custom kernels
    """
    # Use PyTorch's optimized operations with slight epsilon for numerical stability
    norm = x.norm(p=2, dim=-1, keepdim=True)
    return x / (norm + 1e-8)

def replacement_func():
    """
    Returns the optimized L2 normalization function
    """
    return optimized_l2_norm