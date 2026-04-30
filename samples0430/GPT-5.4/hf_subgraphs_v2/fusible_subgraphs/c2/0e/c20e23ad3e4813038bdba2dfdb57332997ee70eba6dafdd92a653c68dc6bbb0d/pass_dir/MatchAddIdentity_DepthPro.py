import torch
import triton
import triton.language as tl
from pass_dir.shared_depthpro_dispatch import depthpro_shared_dispatch, ROUTE_ADD


def pattern(x, y):
    z = x + y
    return z


def replacement_args(x, y):
    return (x, y, ROUTE_ADD)


@triton.jit
def _dummy_kernel(x_ptr, y_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))


def replacement_func():
    return depthpro_shared_dispatch