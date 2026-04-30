import torch
import sys
import os

# Add pass_dir to path so we can import shared_kernels
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_kernels import _dispatch_wrapper


def pattern(in_0, ref):
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(ref)
    return tmp_3


def replacement_args(in_0, ref):
    return (in_0, ref, "expand")


def replacement_func():
    return _dispatch_wrapper