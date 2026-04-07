import torch

# Pattern matching function
def pattern(x):
    return x.reshape(1, 361, 49)

# Simple optimized implementation
@torch.fx.wrap
def optimized_reshape(x):
    batch_size, h, w = x.shape
    return x.reshape(batch_size, h * w // 49, 49)  # Ensure correct reshape

# Argument extraction function
def replacement_args(x):
    return (x,)

# Replacement function
def replacement_func():
    return optimized_reshape