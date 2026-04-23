import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import shared_dispatch


def pattern(in_2):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    return tmp_2


def replacement_args(in_2):
    return (in_2, 'identity')


@triton.jit
def _marker_kernel(x_ptr):
    pass


def replacement_func():
    return shared_dispatch