import os
import sys
import torch
import triton
import triton.language as tl

_PASS_DIR = os.path.dirname(__file__)
if _PASS_DIR not in sys.path:
    sys.path.append(_PASS_DIR)

from shared_kernels import replacement_func


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = in_2 * linear
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "fused_linear_mul")