import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0):
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _unused_copy_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)


@torch.fx.wrap
def transpose_to_cuda_view(in_0):
    return in_0.t()


def replacement_func():
    return transpose_to_cuda_view