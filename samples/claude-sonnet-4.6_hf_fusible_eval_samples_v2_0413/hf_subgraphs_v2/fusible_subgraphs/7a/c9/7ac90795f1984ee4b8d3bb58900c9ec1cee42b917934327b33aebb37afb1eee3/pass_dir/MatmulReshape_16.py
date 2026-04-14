import torch
from pass_dir.shared_dispatch import dispatch


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_16")


def replacement_func():
    return dispatch