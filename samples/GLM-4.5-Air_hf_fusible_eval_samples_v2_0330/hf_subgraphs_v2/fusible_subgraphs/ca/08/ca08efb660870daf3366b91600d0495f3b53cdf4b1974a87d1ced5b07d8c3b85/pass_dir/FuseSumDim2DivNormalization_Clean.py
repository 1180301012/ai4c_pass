import torch

# Pattern matching function
def pattern(in_1):
    """
    Matches the normalization pattern: sum(dim=2, keepdim=True) followed by division
    This matches exactly: tmp_0 = in_1.sum(dim = 2, keepdim = True); tmp_1 = in_1 / tmp_0
    """
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized normalization using direct computation
@torch.fx.wrap
def fused_normalization(in_1):
    """
    Fused normalization combining sum and division in one operation
    """
    # Direct computation without intermediate variables
    return in_1 / in_1.sum(dim=2, keepdim=True)

# Replacement function (no arguments)
def replacement_func():
    return fused_normalization