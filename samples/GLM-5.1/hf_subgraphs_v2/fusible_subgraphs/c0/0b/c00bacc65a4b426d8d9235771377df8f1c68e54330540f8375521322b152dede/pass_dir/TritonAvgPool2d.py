import torch
import triton
import triton.language as tl
import sys
import os

# Add pass_dir to path for shared module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kernels import dispatch_wrapper


def pattern(input):
    result = torch.nn.functional.avg_pool2d(input, 2, 2, 0, True, False, None)
    return result


def replacement_args(input):
    return (input, "avgpool")


def replacement_func():
    return dispatch_wrapper