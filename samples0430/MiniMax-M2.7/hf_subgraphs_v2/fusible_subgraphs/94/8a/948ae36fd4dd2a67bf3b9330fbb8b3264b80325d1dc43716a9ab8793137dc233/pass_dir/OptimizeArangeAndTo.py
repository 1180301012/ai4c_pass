import torch
from torch import device


def pattern(in_0):
    """
    Match the pattern: in_0.to(device=device(...), dtype=torch.bool)
    The pattern matches the full to() operation with both device and dtype kwargs
    """
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2


def replacement_args(in_0):
    """Extract arguments needed for replacement"""
    return (in_0,)


def replacement_func():
    """Return the optimized function"""
    return optimized_to_impl


def optimized_to_impl(input_tensor):
    """Optimized to(bool) implementation.
    Uses simple dtype conversion, letting PyTorch handle the details.
    """
    return input_tensor.to(dtype=torch.bool)