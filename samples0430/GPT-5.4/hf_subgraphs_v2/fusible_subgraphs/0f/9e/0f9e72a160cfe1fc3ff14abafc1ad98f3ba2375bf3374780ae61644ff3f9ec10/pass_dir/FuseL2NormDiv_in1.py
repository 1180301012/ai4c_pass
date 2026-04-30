import torch
import triton
import triton.language as tl
from pass_dir.shared_xclip_norm_kernels import replacement_func


def pattern(in_1):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    return tmp_2


def replacement_args(in_1):
    return (in_1, in_1, "norm_only")