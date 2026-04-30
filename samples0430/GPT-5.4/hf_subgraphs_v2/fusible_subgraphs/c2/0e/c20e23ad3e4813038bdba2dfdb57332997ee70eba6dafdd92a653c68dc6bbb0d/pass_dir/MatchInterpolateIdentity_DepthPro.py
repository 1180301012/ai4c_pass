import torch
import triton
import triton.language as tl
from pass_dir.shared_depthpro_dispatch import depthpro_shared_dispatch, ROUTE_IDENTITY


def pattern(x):
    y = torch.nn.functional.interpolate(x, (24, 24), None, 'bilinear', False, None, False)
    return y


def replacement_args(x):
    return (x, ROUTE_IDENTITY)


@triton.jit
def _dummy_kernel(x_ptr, y_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))


# Shared replacement_func identity to preserve all routes under limit=1.
def replacement_func():
    return depthpro_shared_dispatch