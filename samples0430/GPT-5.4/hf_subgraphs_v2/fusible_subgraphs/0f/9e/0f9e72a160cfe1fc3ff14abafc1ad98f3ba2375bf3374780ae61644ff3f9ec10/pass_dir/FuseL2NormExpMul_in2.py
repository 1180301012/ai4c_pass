import torch
import triton
import triton.language as tl
from pass_dir.shared_xclip_norm_kernels import replacement_func


def pattern(in_2):
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    return tmp_4


def replacement_args(in_2):
    return (in_2, in_2, "norm_only")