import torch
import triton
import triton.language as tl
from pass_dir.shared_linear_bias_kernel import shared_replacement_func


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(in_0.dtype)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dropout_to_linear_rect")


def replacement_func():
    return shared_replacement_func()