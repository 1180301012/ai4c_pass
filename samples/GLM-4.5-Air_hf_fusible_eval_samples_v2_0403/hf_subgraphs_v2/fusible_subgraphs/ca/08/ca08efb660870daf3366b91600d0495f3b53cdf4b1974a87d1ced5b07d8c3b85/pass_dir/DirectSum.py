import torch

# Pattern matching function - match the sum operation
def pattern(in_1):
    tmp_0 = in_1.sum(dim = 2, keepdim = True)
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Direct computation without wrapper overhead
def direct_sum_optimized(in_1):
    """Direct computation without torch.fx.wrap"""
    # Use PyTorch's native sum which is already optimized
    return in_1.sum(dim=2, keepdim=True)

# Replacement function - return function reference
def replacement_func():
    return direct_sum_optimized