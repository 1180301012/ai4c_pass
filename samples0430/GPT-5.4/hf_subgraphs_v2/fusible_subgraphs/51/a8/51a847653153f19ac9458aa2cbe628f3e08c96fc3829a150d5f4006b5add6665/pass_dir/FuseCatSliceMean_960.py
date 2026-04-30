import torch
import triton
import triton.language as tl

from pass_dir.cat_slice_mean_shared import dispatch_cat_or_mean


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None))]
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1, 'cat')


def replacement_func():
    return dispatch_cat_or_mean