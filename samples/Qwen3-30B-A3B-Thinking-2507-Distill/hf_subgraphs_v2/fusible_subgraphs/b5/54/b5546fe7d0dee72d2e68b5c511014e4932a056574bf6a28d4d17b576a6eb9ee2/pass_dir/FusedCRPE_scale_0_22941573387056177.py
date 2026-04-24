"""
Pass: fuse CRPE post-conv operations with scale = 0.22941573387056177
Matches graphs from: coat_tiny (57 channels)
  - bfloat16 / 9  (H=W=14 → reshape(1,8,19,196))
  - bfloat16 / 2  (H=W=28 → reshape(1,8,19,784))
  - bfloat16 / 9  (H=W=56 → reshape(1,8,19,3136))
  - float32 / 2   (H=W=28 → reshape(1,8,19,784))
  - float32 / 9   (H=W=7  → reshape(1,8,19,49))
"""

import torch
from pass_dir.crpe_fused_impl import crpe_fused

_SCALE = 0.22941573387056177
_ROUTE = "crpe_0_2294"


def pattern(in_2, in_3, conv2d, in_4, in_6):
    tmp_3 = torch.cat([in_2, in_3, conv2d], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, 19, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = _SCALE * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 152)
    return tmp_11


def replacement_args(in_2, in_3, conv2d, in_4, in_6):
    return (in_2, in_3, conv2d, in_4, in_6, _SCALE, _ROUTE)


def replacement_func():
    return crpe_fused