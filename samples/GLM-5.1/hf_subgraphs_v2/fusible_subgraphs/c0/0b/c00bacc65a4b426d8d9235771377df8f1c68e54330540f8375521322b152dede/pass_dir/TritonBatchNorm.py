import torch
import triton
import triton.language as tl
import sys
import os

# Add pass_dir to path for shared module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kernels import dispatch_wrapper


def pattern(input, running_mean, running_var, weight, bias):
    result = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return result


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias, "bn")


def replacement_func():
    return dispatch_wrapper