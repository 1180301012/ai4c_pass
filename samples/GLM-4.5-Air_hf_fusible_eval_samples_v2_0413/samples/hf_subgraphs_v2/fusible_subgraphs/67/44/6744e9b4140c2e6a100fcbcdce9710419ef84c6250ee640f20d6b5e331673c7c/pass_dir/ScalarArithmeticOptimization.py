import torch

# Pattern matching function for scalar arithmetic optimization
def pattern(in_0_tensor):
    """Match the pattern: in_0 // constant + 1 (via sym_sum)"""
    tmp_2 = in_0_tensor // 16
    tmp_3 = torch.sym_sum([1, tmp_2])
    return tmp_2, tmp_3

# Argument extraction function
def replacement_args(in_0_tensor):
    return (in_0_tensor,)

# Optimized scalar arithmetic function
@torch.fx.wrap
def optimized_scalar_arithmetic(in_0_tensor):
    """Optimized version: in_0 // 16 + 1"""
    # Since in_0 is a scalar tensor, this becomes very simple
    return (in_0_tensor // 16), (in_0_tensor // 16 + 1)

# Replacement function (must return a callable function)
def replacement_func():
    return optimized_scalar_arithmetic