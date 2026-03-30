import torch

# Pattern matching function - minimal change to match the pattern
def pattern(in_1):
    """
    Minimal pattern matching for normalization
    """
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Minimal optimization - just return the direct computation
@torch.fx.wrap
def minimal_optimization(in_1):
    """
    Minimal optimization - direct computation without any overhead
    """
    # Single expression - minimal variable assignments
    return in_1 / in_1.sum(dim=2, keepdim=True)

# Replacement function
def replacement_func():
    return minimal_optimization