import torch

# Pattern matching function for divisor 32
def pattern(in_0, in_1):
    """Match computation with divisor 32"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // 32
    tmp_2 = torch.sym_sum([1, tmp_1])
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, 32)  # Pass the divisor value

# Optimized function for division + sum with divisor 32
@torch.fx.wrap
def optimized_div_sum_div32(in_0, _):
    """
    Directly compute 1 + (in_0 // 32) more efficiently
    than using torch.sym_sum([1, tmp_1])
    """
    return 1 + (in_0 // 32)

# Replacement function
def replacement_func():
    return optimized_div_sum_div32