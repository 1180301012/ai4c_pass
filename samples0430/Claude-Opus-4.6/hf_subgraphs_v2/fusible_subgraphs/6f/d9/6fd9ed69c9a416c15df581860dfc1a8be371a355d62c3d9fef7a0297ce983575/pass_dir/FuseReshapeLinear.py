import torch
from pass_dir.shared_kernel import shared_dispatch


def pattern(in_4, in_3, in_2):
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    return linear_1


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2, "route_linear2")


def replacement_func():
    return shared_dispatch