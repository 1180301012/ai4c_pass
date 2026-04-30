import torch
import triton
import triton.language as tl
from pass_dir.shared_depthpro_dispatch import depthpro_shared_dispatch, ROUTE_CONV_RELU_ADD


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    tmp_4 = in_2 + tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, ROUTE_CONV_RELU_ADD)


@triton.jit
def _dummy_kernel(x_ptr, y_ptr, n: tl.constexpr):
    offs = tl.arange(0, n)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs))


def replacement_func():
    return depthpro_shared_dispatch