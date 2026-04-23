import torch
from pass_dir.shared_batched_matmul_transpose import fused_batched_dot_reshape


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 128])
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_128")


def replacement_func():
    return fused_batched_dot_reshape