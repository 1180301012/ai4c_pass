import torch

@torch.fx.wrap
def fuse_key_states_transform(in_4):
    """Fuse view + transpose for key_states: [1, 1, 512] -> [1, 8, 1, 64]"""
    return in_4.reshape(1, 8, 1, 64)

@torch.fx.wrap
def fuse_linear_transform(linear):
    """Fuse view + transpose for linear output: [1, 1, 512] -> [1, 8, 1, 64]"""
    return linear.reshape(1, 8, 1, 64)

# Pattern matching function for key_states transformation
def pattern_key_states(in_4):
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4

# Pattern matching function for linear transformation  
def pattern_linear(linear):
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    return tmp_6

# Argument extraction function for key_states
def replacement_args_key_states(in_4):
    return (in_4,)

# Argument extraction function for linear
def replacement_args_linear(linear):
    return (linear,)

# Replacement function returns dictionary mapping pattern names to functions
def replacement_func():
    return {
        'key_states': fuse_key_states_transform,
        'linear': fuse_linear_transform
    }