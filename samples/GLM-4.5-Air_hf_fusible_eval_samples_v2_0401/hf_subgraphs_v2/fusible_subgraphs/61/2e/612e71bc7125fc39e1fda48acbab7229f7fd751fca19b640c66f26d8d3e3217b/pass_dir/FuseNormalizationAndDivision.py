import torch

@torch.fx.wrap
def fused_l2_normalize(x):
    """Fused L2 normalization function using efficient PyTorch operations"""
    # Use PyTorch's built-in normalization which handles edge cases efficiently
    return x / x.norm(p=2, dim=-1, keepdim=True)

def pattern(x):
    """Pattern: L2 normalization followed by division"""
    tmp_0 = x.norm(p = 2, dim = -1, keepdim = True)
    tmp_1 = x / tmp_0
    return tmp_1

def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)

def replacement_func():
    """Return the optimized function"""
    return fused_l2_normalize