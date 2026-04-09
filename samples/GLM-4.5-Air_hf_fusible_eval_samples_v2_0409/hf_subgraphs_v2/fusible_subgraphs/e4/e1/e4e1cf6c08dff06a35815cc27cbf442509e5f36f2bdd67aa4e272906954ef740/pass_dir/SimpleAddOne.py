import torch

# Simple pattern matching for addition operation
def pattern(in_0, in_1, in_2):
    tmp = in_1.float()
    result = 1.0 + tmp
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple replacement function
@torch.fx.wrap
def simple_add_one(in_0, in_1, in_2):
    tmp = in_1.float()
    result = 1.0 + tmp
    return result

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_add_one