import torch
import triton
import triton.language as tl
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_dispatch import unfold_window_dispatch   # noqa: E402


# Pattern: transpose + reshape(1,-1,16,9) + torch.reshape([-1,8,9])
# in_0 = unfold output [1, C*9, S] where C=16, H=8
def pattern(in_0):
    tmp_3 = in_0.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "c16_h8")


def replacement_func():
    return unfold_window_dispatch