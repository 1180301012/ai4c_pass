"""
Optimization pass: fuse position_ids unsqueeze + float conversions into one Triton kernel.
Uses shared dispatch_wrapper (route="position_ids") to satisfy replacement_func_limit.

Pattern matched:
    tmp_19 = in_3[:, None, :]   # unsqueeze to [batch, 1, seq]
    tmp_20 = tmp_19.float()
    tmp_22 = tmp_20.float()     # output [batch, 1, seq] float32
"""

import torch
from pass_dir.dispatch import dispatch_wrapper


def pattern(in_3):
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_22 = tmp_20.float()
    return tmp_22


def replacement_args(in_3):
    # Pass in_3 twice (arg1 is dummy), route="position_ids"
    return (in_3, in_3, "position_ids")


def replacement_func():
    return dispatch_wrapper