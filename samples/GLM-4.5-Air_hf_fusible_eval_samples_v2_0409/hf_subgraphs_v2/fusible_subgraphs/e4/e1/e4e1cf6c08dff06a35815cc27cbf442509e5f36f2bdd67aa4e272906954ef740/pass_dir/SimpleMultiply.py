import torch

# Simple pattern matching for multiplication operation
def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    # Use in_1 to avoid dead code error
    _ = in_1 * 1.0
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple replacement function that just does the multiplication
@torch.fx.wrap
def simple_multiply(in_0, in_1, in_2):
    return in_0 * in_2

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_multiply