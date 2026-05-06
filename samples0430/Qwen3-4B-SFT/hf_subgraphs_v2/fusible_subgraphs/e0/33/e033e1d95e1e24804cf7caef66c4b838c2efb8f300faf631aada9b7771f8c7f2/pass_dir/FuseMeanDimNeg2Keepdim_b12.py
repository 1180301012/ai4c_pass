import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper  # noqa: F401


def pattern(y):
    """
    Mean over dim=-2 (dim 1 in [B, S, C]) with keepdim=True.
    Applies to all float16, bfloat16, float32 graphs.
    """
    out = y.mean(dim=-2, keepdim=True)
    return out


def replacement_args(y):
    return (y, "mean_dim1")


def replacement_func():
    return dispatch_wrapper