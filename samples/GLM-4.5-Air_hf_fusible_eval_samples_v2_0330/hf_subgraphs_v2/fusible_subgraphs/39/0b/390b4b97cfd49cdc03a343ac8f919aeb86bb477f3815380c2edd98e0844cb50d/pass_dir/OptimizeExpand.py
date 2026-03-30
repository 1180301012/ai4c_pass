import torch

# Pattern matching function - unsqueeze followed by expand
def pattern(tmp_2):
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    return tmp_6

# Argument extraction function
def replacement_args(tmp_2):
    return (tmp_2,)

# Optimized version - direct tensor expansion
@torch.fx.wrap
def optimized_expand(tmp_2):
    # Direct expansion without intermediate unsqueeze and expand operations
    # This is more efficient since it avoids creating temporary tensors
    return tmp_2.expand(3, -1, -1)

# Replacement function
def replacement_func():
    return optimized_expand