import torch
import triton
import triton.language as tl
from pass_dir.shared_xclip_norm_kernels import replacement_func


def pattern(in_0, tmp_4):
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return tmp_6


def replacement_args(in_0, tmp_4):
    return (in_0, tmp_4, "exp_mul")