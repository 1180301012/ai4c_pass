import torch
import triton
import triton.language as tl

from pass_dir.shared_matmul_view import triton_batched_matmul_view


def pattern(in_0, in_1, s0, s1, s2, s3):
    tmp_0 = torch.matmul(in_1, in_0)
    tmp_1 = tmp_0.view(s0, s1, s2, s3)
    return tmp_1


def replacement_args(in_0, in_1, s0, s1, s2, s3):
    return (in_0, in_1, (s0, s1, s2, s3))


def replacement_func():
    return triton_batched_matmul_view