"""
Pass: Fuse matmul + reshape with reshape target [-1, 16].
Matches: matmul(in_1, in_0) → reshape(..., [-1, 16])
"""
import torch
from pass_dir.fused_kernels import fuse_matmul_reshape_16


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fuse_matmul_reshape_16