"""
Optimization pass: replace broadcast elementwise multiply with Triton.

Matches rtmpose-l pattern:
    tmp_3 = in_2 * in_1
    return tmp_3

in_1 : scale  [N]
in_2 : tensor [B*H, N]
out  : [B*H, N]  (scale broadcasts over outer dimensions)

Uses shared fused_dispatch with route="elem".
"""

import torch
from pass_dir.gemm_scale_kernels import fused_dispatch


def pattern(in_2, in_1):
    tmp_3 = in_2 * in_1
    return tmp_3


def replacement_args(in_2, in_1):
    # x=in_2 (big), scale=in_1 (small [N]), ignore arg2 (fused_dispatch pad)
    return (in_2, in_1, in_1, "elem")


def replacement_func():
    return fused_dispatch