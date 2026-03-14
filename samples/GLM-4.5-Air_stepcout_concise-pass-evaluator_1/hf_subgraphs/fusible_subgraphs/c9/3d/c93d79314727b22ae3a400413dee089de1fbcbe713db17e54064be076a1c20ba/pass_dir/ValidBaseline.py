import torch

# Pattern matching function - match matmul exactly as it appears in the computation
def pattern(in_2, in_3):
    """Match torch.matmul operation exactly as it appears in the original"""
    return torch.matmul(in_2, in_3)

# Replacement arguments function
def replacement_args(in_2, in_3):
    """Extract arguments for the pattern replacement"""
    return (in_2, in_3)

# Replacement function - use standard PyTorch operation (completely safe)
def replacement_func():
    """Return standard matmul function - no optimization but valid"""
    return torch.matmul