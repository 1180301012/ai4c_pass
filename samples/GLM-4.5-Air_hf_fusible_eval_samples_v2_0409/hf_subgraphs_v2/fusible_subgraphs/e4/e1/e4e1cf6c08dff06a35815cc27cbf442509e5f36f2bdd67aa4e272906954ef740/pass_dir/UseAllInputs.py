import torch

# Pattern that uses all three inputs
def pattern(x, y, z):
    # Use all three inputs to avoid dead code
    result = x + y * z
    return result

# Argument extraction function
def replacement_args(x, y, z):
    return (x, y, z)

# Replacement function that uses all inputs
@torch.fx.wrap
def use_all_inputs(x, y, z):
    return x + y * z

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return use_all_inputs