import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Match the scalar multiplication pattern: in_1 * scalar
    """
    scalar = 0.3535533905932738
    return in_1 * scalar


def replacement_args(in_1):
    return (in_1,)


# Use a simpler approach - rely on PyTorch's native operations for the scalar mul
# The key is that this pass acts as a placeholder to enable the transpose optimization
# Actually, we just use torch's native operations since they're already well-optimized
@torch.fx.wrap
def scalar_mul_wrapper(in_1):
    scalar = 0.3535533905932738
    return in_1 * scalar


def replacement_func():
    return scalar_mul_wrapper