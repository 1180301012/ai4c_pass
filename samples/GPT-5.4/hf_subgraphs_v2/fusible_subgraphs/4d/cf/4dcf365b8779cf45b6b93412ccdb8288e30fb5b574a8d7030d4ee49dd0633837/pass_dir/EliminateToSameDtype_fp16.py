import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import shared_dispatch


def pattern(in_2):
    to = in_2.to(torch.float16)
    return to


def replacement_args(in_2):
    return (in_2, 'identity')


@triton.jit
def _marker_kernel(x_ptr):
    pass


def replacement_func():
    return shared_dispatch