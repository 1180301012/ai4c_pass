import torch
import sys
import os

# Import shared dispatch wrapper
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from _kernels import replacement_func


def pattern(input):
    pool_out = torch.nn.functional.avg_pool2d(input, 2, 2, 0, True, False, None)
    return pool_out


def replacement_args(input):
    return (input, "avg_pool2d")