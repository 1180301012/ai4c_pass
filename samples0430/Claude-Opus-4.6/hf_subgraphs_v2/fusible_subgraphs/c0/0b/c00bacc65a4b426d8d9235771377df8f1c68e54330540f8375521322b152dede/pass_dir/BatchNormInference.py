import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)


def replacement_func():
    return dispatch_wrapper