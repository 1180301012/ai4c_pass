"""
Fuse: transpose(1,2) → reshape(1,-1,384,9) → reshape(-1,64,9)
for im2col output [1, 3456, L] → final output [L*6, 64, 9]

Uses the shared _dispatch_wrapper (same object as UnfoldSlidingWindow_16_8)
so both passes satisfy the replacement_func_limit and are loaded together.

Covers float16, bfloat16 for YituTech/conv-bert-base.
"""

import torch
from pass_dir._shared_dispatch import _dispatch_wrapper


def pattern(in_0):
    tmp_3 = in_0.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "384_64")


def replacement_func():
    return _dispatch_wrapper