import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import replacement_dispatch


def pattern(tmp_7, in_1, in_0):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-05)
    return tmp_8


def replacement_args(tmp_7, in_1, in_0):
    return (tmp_7, in_1, in_0, "layernorm_lastdim")


def replacement_func():
    return replacement_dispatch