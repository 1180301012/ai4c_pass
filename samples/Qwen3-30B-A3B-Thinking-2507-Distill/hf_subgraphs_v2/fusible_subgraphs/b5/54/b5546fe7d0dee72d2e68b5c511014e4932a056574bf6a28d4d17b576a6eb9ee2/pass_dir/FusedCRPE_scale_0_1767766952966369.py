"""
Pass: fuse CRPE post-conv operations with scale = 0.1767766952966369
Matches graphs from: coat_lite_tiny (96 channels) and coat_lite_medium_384 (96 channels)
  - bfloat16 / 9  (H=W=14 → reshape(1,8,32,196))
  - bfloat16 / 9  (H=W=48 → reshape(1,8,32,2304))
  - float32 / 2   (H=W=28 → reshape(1,8,32,784))
  - float16 / 2   (H=W=7  → reshape(1,8,32, 49))
  - float16 / 9   (H=W=14 → reshape(1,8,32,196))
"""

import torch
from pass_dir.crpe_fused_impl import crpe_fused

_SCALE = 0.1767766952966369
_ROUTE = "crpe_0_1768"


def pattern(in_2, in_3, conv2d, in_4, in_6):
    tmp_3 = torch.cat([in_2, in_3, conv2d], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, 32, 49)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = _SCALE * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 50, 256)
    return tmp_11


def replacement_args(in_2, in_3, conv2d, in_4, in_6):
    return (in_2, in_3, conv2d, in_4, in_6, _SCALE, _ROUTE)


def replacement_func():
    return crpe_fused