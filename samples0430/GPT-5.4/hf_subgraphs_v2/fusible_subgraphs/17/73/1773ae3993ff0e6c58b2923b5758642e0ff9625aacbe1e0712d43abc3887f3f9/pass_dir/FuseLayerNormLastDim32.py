import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import shared_dispatch


def pattern(in_0, in_1, in_4):
    tmp_3 = torch.nn.functional.layer_norm(in_4, (32,), in_1, in_0, 1e-12)
    return tmp_3


def replacement_args(in_0, in_1, in_4):
    return (in_0, in_1, in_4, in_4.shape[1], 32)


def replacement_func():
    return shared_dispatch