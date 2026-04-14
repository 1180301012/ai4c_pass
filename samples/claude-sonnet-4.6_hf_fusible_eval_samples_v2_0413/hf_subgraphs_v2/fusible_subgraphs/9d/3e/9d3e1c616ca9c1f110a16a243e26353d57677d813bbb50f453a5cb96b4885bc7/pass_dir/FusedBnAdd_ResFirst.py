"""
Pass: FusedBnAdd_ResFirst  –  Pattern B

Matches deeppose_resnet_101 start116_end119_1:
  conv2d(in_5, in_4, None, (1,1),(0,0),(1,1),1)
  batch_norm(conv, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
  bn_out += in_6   →   return bn_out
"""

import torch
from pass_dir.shared_kernel import dispatch_conv_bn_add


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv_out = torch.conv2d(in_5, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    bn_out = torch.nn.functional.batch_norm(
        conv_out, in_0, in_1, in_3, in_2, False, 0.1, 1e-05
    )
    result = bn_out.add_(in_6)
    return result


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "B")


def replacement_func():
    return dispatch_conv_bn_add