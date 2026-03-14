import torch

# Pattern matching function - matches flatten(2) followed by permute(0, 2, 1)
def pattern(x):
    tmp_0 = x.flatten(2)
    tmp_1 = tmp_0.permute(0, 2, 1)
    return tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)


@torch.fx.wrap
def flatten_permute_opt(x):
    # Use flatten + mT (matrix transpose on last two dims) - minimal overhead
    return x.flatten(2).mT


# Replacement function - returns the function reference
def replacement_func():
    return flatten_permute_opt