import torch
from pass_dir.shared_kernel import fused_linear_noop_dropout


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_noop_dropout