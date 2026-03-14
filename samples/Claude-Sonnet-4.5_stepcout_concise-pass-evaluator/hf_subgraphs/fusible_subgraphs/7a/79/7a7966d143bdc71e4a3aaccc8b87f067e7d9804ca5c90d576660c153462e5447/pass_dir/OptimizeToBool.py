import torch
from torch import device

def pattern(input_tensor):
    """
    Pattern to match: tensor.to(device=cuda, dtype=torch.bool)
    """
    result = input_tensor.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    """
    Return the actual torch operation directly without wrapping
    """
    # Return a lambda that does the conversion directly
    return lambda x: x.to(dtype=torch.bool)