import os
import sys
import torch
import triton
import triton.language as tl

_PASS_DIR = os.path.dirname(__file__)
if _PASS_DIR not in sys.path:
    sys.path.append(_PASS_DIR)

from shared_kernels import replacement_func


def pattern(in_1, in_2):
    tmp_3 = in_2 * in_1
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2, "broadcast_scale_mul")