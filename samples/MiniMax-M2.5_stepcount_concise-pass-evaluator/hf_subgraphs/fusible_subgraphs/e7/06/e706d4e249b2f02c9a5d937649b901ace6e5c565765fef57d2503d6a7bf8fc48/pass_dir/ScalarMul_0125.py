import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Match the scalar multiplication pattern: in_1 * scalar (0.125)
    """
    scalar = 0.125
    return in_1 * scalar


def replacement_args(in_1):
    return (in_1,)


# Use PyTorch's native operations since they're already well-optimized
@torch.fx.wrap
def scalar_mul_wrapper_125(in_1):
    scalar = 0.125
    return in_1 * scalar


def replacement_func():
    return scalar_mul_wrapper_125