import torch
from pass_dir.fused_linear_kernel import linear_triton_dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "p0_r1")


def replacement_func():
    return linear_triton_dispatch