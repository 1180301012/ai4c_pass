import torch

# Pattern matching function - just match matmul without optimization
def pattern(in_2, in_3):
    """Match torch.matmul operation - identical to original"""
    return torch.matmul(in_2, in_3)

# Replacement arguments function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Replacement function - use original PyTorch matmul (no optimization)
def replacement_func():
    return torch.matmul