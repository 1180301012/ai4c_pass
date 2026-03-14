import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Match just the scalar multiplication pattern.
    Returns a single value.
    """
    scalar = 0.125
    return in_1 * scalar


def replacement_args(in_1):
    return (in_1,)


# PyTorch's native operation - already optimized
@torch.fx.wrap
def scalar_mul_0125(in_1):
    return in_1 * 0.125


def replacement_func():
    return scalar_mul_0125