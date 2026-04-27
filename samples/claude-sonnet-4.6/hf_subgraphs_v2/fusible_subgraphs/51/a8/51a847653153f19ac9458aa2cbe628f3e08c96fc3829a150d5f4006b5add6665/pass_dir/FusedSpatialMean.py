import torch
from pass_dir.shared_cat_mean_kernel import triton_spatial_mean


def pattern(x):
    tmp_2 = x.mean((2, 3), keepdim=True)
    return tmp_2


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_spatial_mean