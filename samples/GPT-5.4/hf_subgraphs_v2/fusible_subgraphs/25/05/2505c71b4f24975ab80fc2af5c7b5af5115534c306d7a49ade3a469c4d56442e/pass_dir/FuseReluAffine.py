import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "relu_affine")


def replacement_func():
    return replacement_impl