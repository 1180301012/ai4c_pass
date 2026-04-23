import os
import sys
import torch

_pass_dir = os.path.dirname(__file__)
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)

from shared_maxpool_replacement import triton_max_pool2d_3x3_s2_p1


def pattern(x):
    out = torch.nn.functional.max_pool2d(x, 3, 2, 1)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_max_pool2d_3x3_s2_p1