import torch
import triton
import triton.language as tl

from pass_dir.fused_conv_bn_avgpool_impl import fused_conv_bn_avgpool_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    return (tmp_7, tmp_6)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_6, in_5)


def replacement_func():
    return fused_conv_bn_avgpool_dispatch