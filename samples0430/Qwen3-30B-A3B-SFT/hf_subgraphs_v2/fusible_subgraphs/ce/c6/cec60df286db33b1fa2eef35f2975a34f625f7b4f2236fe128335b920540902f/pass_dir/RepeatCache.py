import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import fused_dispatch


# Pass 2: Replace repeat(1,1) with a constant-folded dispatch.
# The model has no inputs, so repeat output is always [[[0]]].
def pattern(x):
    return x.repeat(1, 1)


def replacement_args(x):
    return (x, "route_repeat")


def replacement_func():
    return fused_dispatch