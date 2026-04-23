import torch
import triton
import triton.language as tl

from pass_dir.fused_tail_common import fused_tail_dispatch


SCALE = 0.125
ROUTE = "scale_0125"


def pattern(tmp_7, in_4):
    tmp_8 = 0.125 * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    return tmp_10


def replacement_args(tmp_7, in_4):
    return (tmp_7, in_4, SCALE, ROUTE)


def replacement_func():
    return fused_tail_dispatch