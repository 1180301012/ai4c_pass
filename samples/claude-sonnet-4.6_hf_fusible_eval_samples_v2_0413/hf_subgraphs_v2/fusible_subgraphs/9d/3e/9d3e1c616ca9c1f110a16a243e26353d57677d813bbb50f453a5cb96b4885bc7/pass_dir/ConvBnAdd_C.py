"""
Pass: ConvBnAdd_C  –  Pattern C

Matches resnet10t.c3_in1k start23_end26_7:
  conv2d(in_5, in_0, None, (1,1),(0,0),(1,1),1)
  batch_norm(conv, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
  in_6 += bn_out   →   return in_6
"""

import torch
from pass_dir.shared_kernel import dispatch_conv_bn_add


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv_out = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    bn_out = torch.nn.functional.batch_norm(
        conv_out, in_1, in_2, in_4, in_3, False, 0.1, 1e-05
    )
    result = in_6.add_(bn_out)
    return result


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "C")


def replacement_func():
    return dispatch_conv_bn_add