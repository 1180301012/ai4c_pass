import torch
from pass_dir.relu_shared import relu_dispatch


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.flatten(tmp_0, 1, -1)
    return tmp_1


def replacement_args(in_0):
    return (in_0, "inplace")


def replacement_func():
    return relu_dispatch