import os
import sys
import torch
import triton
import triton.language as tl

_pass_dir = os.path.dirname(__file__)
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)

from shared_decode_head_only import decode_head_only


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    tmp_11 = torch.nn.functional.interpolate(conv2d, size=(512, 512), mode='bilinear', align_corners=False)
    conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
    tmp_14 = torch.nn.functional.relu(tmp_13, inplace=False)
    to = tmp_14.to(torch.bfloat16)
    conv2d_2 = torch.conv2d(to, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_16 = torch.nn.functional.interpolate(conv2d_2, size=(512, 512), mode='bilinear', align_corners=False)
    return (tmp_11,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
    return (in_10, in_8, in_7)


def replacement_func():
    return decode_head_only