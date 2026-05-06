import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import fused_view_repeat_dispatch


# Pattern: match the view(1,-1) + repeat(2,1) subgraph.
# x here is the arange(0,128) result (arange itself is not matched).
# x.view(1,-1) reshapes (128,) → (1,128); repeat(2,1) expands → (2,128).
def pattern(x):
    tmp_1 = x.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args(x):
    return (x, "route_128")


def replacement_func():
    return fused_view_repeat_dispatch