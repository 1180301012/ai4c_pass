import torch
import triton
from pass_dir.conv1x1_shared import run_conv1x1_slice


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.compiler.disable
@torch.fx.wrap
def _conv_s2_sf_512(in_0, in_1):
    return run_conv1x1_slice(in_0, in_1, 2, 512, True)


def replacement_func():
    return _conv_s2_sf_512