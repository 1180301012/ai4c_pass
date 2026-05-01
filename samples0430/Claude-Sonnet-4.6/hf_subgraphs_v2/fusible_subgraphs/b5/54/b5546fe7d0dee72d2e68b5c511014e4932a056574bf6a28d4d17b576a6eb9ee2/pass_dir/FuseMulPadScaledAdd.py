"""
Fallback pass: fuses transpose(-1,-2) + element-wise-mul.
Uses the shared coat_dispatch wrapper so it counts as the same
replacement_func as the conv+cat fusion passes.
"""

import torch
import triton
from pass_dir.coat_dispatch import coat_dispatch


def pattern(x, in_6):
    tmp_5 = x.transpose(-1, -2)
    return in_6 * tmp_5


def replacement_args(x, in_6):
    # Pad to 6 tensor args + route string for the shared dispatch signature
    return (x, in_6, None, None, None, None, "transpose")


def replacement_func():
    return coat_dispatch