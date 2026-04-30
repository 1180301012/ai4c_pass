import torch
from pass_dir._matmul_kernels import batched_matmul_triton


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul


def replacement_args(in_0, in_1):
    return (in_1, in_0)


def replacement_func():
    return batched_matmul_triton