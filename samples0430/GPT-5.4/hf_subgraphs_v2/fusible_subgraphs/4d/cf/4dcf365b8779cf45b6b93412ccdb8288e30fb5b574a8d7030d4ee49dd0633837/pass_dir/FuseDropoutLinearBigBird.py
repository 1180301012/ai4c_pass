import torch
import triton
import triton.language as tl
from pass_dir.shared_linear_bias_kernel import shared_replacement_func


def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dropout_linear_bigbird")


def replacement_func():
    return shared_replacement_func()