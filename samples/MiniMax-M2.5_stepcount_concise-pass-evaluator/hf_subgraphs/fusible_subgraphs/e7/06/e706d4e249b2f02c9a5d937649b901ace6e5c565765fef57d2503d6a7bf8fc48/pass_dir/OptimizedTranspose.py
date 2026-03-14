import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    - tmp_0 = in_1 * scalar
    - tmp_1 = in_0.transpose(-1, -2)
    Returns both results.
    """
    scalar = 0.3535533905932738
    tmp_0 = in_1 * scalar
    tmp_1 = in_0.transpose(-1, -2)
    return (tmp_0, tmp_1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Use PyTorch's native operations - they're already highly optimized
@torch.fx.wrap
def fused_ops(in_0, in_1):
    scalar = 0.3535533905932738
    # Use in_1 * scalar (same as original)
    out_0 = in_1 * scalar
    # Use in_0.transpose(-1, -2) (same as original)
    out_1 = in_0.transpose(-1, -2)
    return out_0, out_1


def replacement_func():
    return fused_ops