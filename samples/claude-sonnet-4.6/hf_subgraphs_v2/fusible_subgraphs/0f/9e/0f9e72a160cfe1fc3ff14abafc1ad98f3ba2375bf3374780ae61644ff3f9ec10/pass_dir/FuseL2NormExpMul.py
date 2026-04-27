import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(scale, x):
    tmp_3 = x.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = x / tmp_3
    tmp_5 = scale.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4)


def replacement_args(scale, x):
    return (scale, x, "l2norm_exp_mul")


def replacement_func():
    return shared_dispatch