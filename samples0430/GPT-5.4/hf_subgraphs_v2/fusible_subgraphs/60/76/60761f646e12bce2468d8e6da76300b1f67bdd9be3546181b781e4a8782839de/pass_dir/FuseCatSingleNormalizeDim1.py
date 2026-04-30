import os
import sys
import torch
import triton
import triton.language as tl

_pass_dir = os.path.dirname(__file__)
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)

from shared_triton_l2_normalize_dim1 import triton_l2_normalize_dim1


def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_l2_normalize_dim1