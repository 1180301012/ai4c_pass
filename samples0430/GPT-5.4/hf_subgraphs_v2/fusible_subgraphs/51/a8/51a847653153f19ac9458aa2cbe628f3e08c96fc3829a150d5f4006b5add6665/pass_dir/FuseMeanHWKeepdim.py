import torch
import triton
import triton.language as tl

from pass_dir.cat_slice_mean_shared import dispatch_cat_or_mean


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x, x, 'mean')


def replacement_func():
    return dispatch_cat_or_mean