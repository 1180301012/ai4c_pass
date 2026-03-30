import torch

# Pattern matching function - cumsum followed by subtraction
def pattern(in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    return tmp_2

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized version - efficient cumsum and subtraction
@torch.fx.wrap
def optimized_cumsum_sub(in_1):
    # For cumsum operations, PyTorch is already highly optimized
    # Just ensure we do cumsum followed by -1 efficiently
    return in_1.cumsum(-1) - 1

# Replacement function
def replacement_func():
    return optimized_cumsum_sub