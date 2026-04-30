import torch
import sys
import os

# Add pass_dir to path so we can import shared_kernels
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_kernels import _triton_broadcast_mul


def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


def replacement_func():
    return _triton_broadcast_mul