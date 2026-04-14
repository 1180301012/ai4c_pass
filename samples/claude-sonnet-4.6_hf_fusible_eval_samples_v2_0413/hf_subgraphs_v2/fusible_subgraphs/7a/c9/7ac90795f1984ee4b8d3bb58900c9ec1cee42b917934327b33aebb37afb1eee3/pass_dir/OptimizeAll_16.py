import torch
from pass_dir.shared_dispatch_all import dispatch_all


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    tmp_2 = in_2.transpose(-1, -2)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_16")


def replacement_func():
    return dispatch_all