import torch
from pass_dir.shared_kernels import triton_matmul_reshape


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return torch.reshape(matmul, [-1, 16])


def replacement_args(in_0, in_1):
    return (in_0, in_1, "16")


def replacement_func():
    return triton_matmul_reshape