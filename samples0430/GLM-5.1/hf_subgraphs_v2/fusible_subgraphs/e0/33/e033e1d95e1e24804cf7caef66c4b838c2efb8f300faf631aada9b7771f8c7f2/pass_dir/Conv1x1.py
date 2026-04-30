import torch
import sys
import os

# Import shared dispatch wrapper to satisfy replacement_func_limit
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from _shared import dispatch_wrapper


def pattern(bias, weight, input):
    return torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(bias, weight, input):
    return (bias, weight, input, "conv1x1")


def replacement_func():
    return dispatch_wrapper