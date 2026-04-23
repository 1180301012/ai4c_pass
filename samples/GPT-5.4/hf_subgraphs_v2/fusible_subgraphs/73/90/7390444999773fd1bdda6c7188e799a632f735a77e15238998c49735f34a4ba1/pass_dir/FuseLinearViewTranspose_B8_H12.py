import torch
from pass_dir.shared_fused_linear_view_transpose import replacement_args_shared, replacement_func_shared


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(8, -1, 12, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4


def replacement_args(in_0, in_1, in_3):
    return replacement_args_shared(in_0, in_1, in_3)


def replacement_func():
    return replacement_func_shared()