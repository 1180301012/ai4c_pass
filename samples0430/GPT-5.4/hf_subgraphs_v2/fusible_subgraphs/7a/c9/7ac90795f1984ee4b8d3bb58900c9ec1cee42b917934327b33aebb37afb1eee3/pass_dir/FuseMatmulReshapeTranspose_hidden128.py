import torch
import triton
import triton.language as tl

from pass_dir.shared_smallk_matmul_reshape import smallk_matmul_reshape


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 128])
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1, 128)


def replacement_func():
    return smallk_matmul_reshape