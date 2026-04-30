import torch
import sys
import os

# Import shared dispatch wrapper to satisfy replacement_func_limit
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from _shared import dispatch_wrapper


def pattern(x):
    return x.mean(dim=-2, keepdim=True)


def replacement_args(x):
    return (x, "mean_neg2_keepdim")


def replacement_func():
    return dispatch_wrapper