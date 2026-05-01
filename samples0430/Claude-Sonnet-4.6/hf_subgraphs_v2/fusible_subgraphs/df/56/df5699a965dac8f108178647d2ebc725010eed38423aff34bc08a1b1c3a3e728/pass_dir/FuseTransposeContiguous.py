import torch
from pass_dir.shared_kernels import dispatch


def pattern(x):
    """
    Matches any transpose(1, 2) followed by contiguous().
    This fires for BOTH path1 and path2, for ALL batch sizes.
    The input x is the result of view(B, 2, C, H, W) – a contiguous 5-D tensor.
    """
    t   = torch.transpose(x, 1, 2)
    out = t.contiguous()
    return out


def replacement_args(x):
    # route "sc" → _strided_copy_impl(x)
    return (x, x, x, x, "sc")


def replacement_func():
    return dispatch