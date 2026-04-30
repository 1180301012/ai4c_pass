import torch

from pass_dir.shared_kernels import fused_dispatch


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1, in_2):
    # Rearrange: kernel computes arg0 + arg1 + arg2 = in_1 + in_2 + in_0
    return (in_1, in_2, in_0, "add_add_mean")


def replacement_func():
    return fused_dispatch