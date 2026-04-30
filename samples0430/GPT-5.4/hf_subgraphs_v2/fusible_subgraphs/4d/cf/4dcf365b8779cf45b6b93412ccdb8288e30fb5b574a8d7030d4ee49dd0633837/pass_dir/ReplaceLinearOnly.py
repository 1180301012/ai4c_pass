import torch
import triton
import triton.language as tl
from pass_dir.shared_linear_bias_kernel import shared_replacement_func


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "linear_only")


def replacement_func():
    return shared_replacement_func()