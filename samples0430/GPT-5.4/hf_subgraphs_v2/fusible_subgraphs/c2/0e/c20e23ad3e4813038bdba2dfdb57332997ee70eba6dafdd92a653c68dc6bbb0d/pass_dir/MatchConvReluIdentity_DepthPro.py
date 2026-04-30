import torch
import triton
import triton.language as tl
from pass_dir.shared_depthpro_dispatch import depthpro_shared_dispatch, ROUTE_CONV_RELU


def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3, in_3, ROUTE_CONV_RELU)


@triton.jit
def _dummy_kernel(x_ptr, y_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))


def replacement_func():
    return depthpro_shared_dispatch