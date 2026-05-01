"""
Optimization pass: fuse inv_freq unsqueeze + float conversions + expand into one Triton kernel.
Uses shared dispatch_wrapper (route="inv_freq") to satisfy replacement_func_limit.

Pattern matched:
    tmp_15 = in_1[None, :, None]          # unsqueeze to [1, freq, 1]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device('cuda', 0))
    tmp_21 = tmp_18.float()               # output [1, freq, 1] float32
"""

import torch
from torch import device
from pass_dir.dispatch import dispatch_wrapper


def pattern(in_1):
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_21 = tmp_18.float()
    return tmp_21


def replacement_args(in_1):
    # Pass in_1 twice (arg1 is dummy), route="inv_freq"
    return (in_1, in_1, "inv_freq")


def replacement_func():
    return dispatch_wrapper