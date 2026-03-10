import torch

def pattern(in_1, constant):
    tmp_0 = in_1 * constant
    return tmp_0

def replacement_args(in_1, constant):
    return (in_1, constant)

@torch.fx.wrap
def optimized_scalar_multiply(x, scalar):
    """
    Optimized scalar multiplication that leverages PyTorch's native optimizations.
    
    For the specific constants used in these models, we can return the operation
    as-is since PyTorch's backend is already highly optimized for scalar multiplication.
    """
    # For these specific constants, just use the native PyTorch operation
    # which will be optimized by the backend compiler
    return x * scalar

def replacement_func():
    return optimized_scalar_multiply