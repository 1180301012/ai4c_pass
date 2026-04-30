import torch
from pass_dir.shared_dispatch import shared_replacement


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "tiny_swapped_p0_0")


def replacement_func():
    return shared_replacement