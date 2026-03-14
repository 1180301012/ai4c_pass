import torch
import triton
import triton.language as tl
from torch import device


# Pattern matching function - matches the computation: in_0 / 8.0 + in_1
def pattern(in_0, in_1):
    tmp_0 = in_0 / 8.0
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = tmp_0 + tmp_1
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Minimal optimized implementation - direct PyTorch operations
# The key optimization is using multiply by reciprocal instead of division
@torch.fx.wrap
def fused_div_add_kernel_wrapper(in_0, in_1):
    # Direct computation with multiply by reciprocal
    return in_0 * 0.125 + in_1


def replacement_func():
    return fused_div_add_kernel_wrapper