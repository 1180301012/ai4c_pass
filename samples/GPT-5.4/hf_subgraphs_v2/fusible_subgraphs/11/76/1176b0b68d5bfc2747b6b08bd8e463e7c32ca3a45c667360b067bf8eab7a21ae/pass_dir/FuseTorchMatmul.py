import torch
from pass_dir.shared_batched_matmul import replacement_dispatch


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return replacement_dispatch