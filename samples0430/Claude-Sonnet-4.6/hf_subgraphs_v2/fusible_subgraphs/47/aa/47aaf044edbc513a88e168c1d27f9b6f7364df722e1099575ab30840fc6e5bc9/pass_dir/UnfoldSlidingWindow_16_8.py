"""
Fuse: transpose(1,2) → reshape(1,-1,16,9) → reshape(-1,8,9)
for im2col output [1, 144, L] → final output [L*2, 8, 9]

Uses the shared _dispatch_wrapper (same object as UnfoldSlidingWindow_384_64)
so both passes satisfy the replacement_func_limit and are loaded together.

Covers float16, bfloat16, float32 for tiny-random-ConvBertForSequenceClassification.
"""

import torch
from pass_dir._shared_dispatch import _dispatch_wrapper


def pattern(in_0):
    tmp_3 = in_0.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "16_8")


def replacement_func():
    return _dispatch_wrapper