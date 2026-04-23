import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import shared_dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return (linear,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'linear')


@triton.jit
def _marker_kernel(x_ptr):
    pass


def replacement_func():
    return shared_dispatch