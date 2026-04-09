import torch

# Pattern matching function
def pattern(in_0, divisor):
    """Match division followed by sum with scalar 1"""
    tmp_1 = in_0 // divisor
    tmp_2 = torch.sym_sum([1, tmp_1])
    return tmp_2

# Argument extraction function
def replacement_args(in_0, divisor):
    return (in_0, divisor)

# Optimized function for division + sum
@torch.fx.wrap
def optimized_div_sum(in_0, divisor):
    """
    Directly compute 1 + (in_0 // divisor) more efficiently
    than using torch.sym_sum([1, tmp_1])
    """
    return 1 + (in_0 // divisor)

# Replacement function
def replacement_func():
    return optimized_div_sum