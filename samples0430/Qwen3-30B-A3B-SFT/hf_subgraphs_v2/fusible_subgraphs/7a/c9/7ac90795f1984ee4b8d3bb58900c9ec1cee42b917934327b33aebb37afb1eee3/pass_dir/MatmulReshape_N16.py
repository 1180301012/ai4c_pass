"""
Pass: fuse matmul(in_1, in_0) + reshape([-1, 16]) for ConvBERT tiny model.
  in_0: [B, 9, 1], in_1: [B, 8, 9]  →  out: [-1, 16]
"""
import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(a, b):
    matmul = torch.matmul(a, b)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    return tmp_1


def replacement_args(a, b):
    # c = b (dummy placeholder for shared_dispatch signature); route selects branch
    return (a, b, b, "matmul_reshape")


def replacement_func():
    return shared_dispatch