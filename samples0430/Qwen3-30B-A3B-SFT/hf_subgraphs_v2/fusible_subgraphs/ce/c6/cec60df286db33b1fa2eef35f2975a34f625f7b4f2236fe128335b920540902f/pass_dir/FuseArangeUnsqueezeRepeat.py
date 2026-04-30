import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import fused_dispatch


# Pass 1: Replace unsqueeze(0) with a constant-folded dispatch.
# The model has no inputs, so unsqueeze output is always [[[0]]].
def pattern(x):
    return x.unsqueeze(0)


def replacement_args(x):
    return (x, "route_unsqueeze")


def replacement_func():
    return fused_dispatch