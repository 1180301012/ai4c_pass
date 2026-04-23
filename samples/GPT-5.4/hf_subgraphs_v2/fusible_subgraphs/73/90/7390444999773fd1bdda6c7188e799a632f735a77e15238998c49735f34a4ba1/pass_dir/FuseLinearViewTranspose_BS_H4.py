import torch
from pass_dir.shared_fused_linear_view_transpose import replacement_args_shared, replacement_func_shared


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(in_3.shape[0], -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return replacement_args_shared(in_0, in_1, in_3)


def replacement_func():
    return replacement_func_shared()