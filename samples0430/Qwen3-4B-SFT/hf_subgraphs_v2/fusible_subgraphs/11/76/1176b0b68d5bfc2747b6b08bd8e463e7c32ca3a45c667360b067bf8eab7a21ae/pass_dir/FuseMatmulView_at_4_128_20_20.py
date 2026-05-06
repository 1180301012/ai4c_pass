import torch
from pass_dir.shared_triton import triton_batched_matmul


def pattern(in_0, in_1):
    matmul = in_1 @ in_0
    tmp_1 = matmul.view(4, 128, 20, 20)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_batched_matmul