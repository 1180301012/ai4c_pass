import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fused_linear_transpose_dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0, "normal_transposed")


def replacement_func():
    return fused_linear_transpose_dispatch