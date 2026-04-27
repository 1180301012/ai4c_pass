import torch
import triton
import triton.language as tl
from pass_dir.coat_shared_kernel import coat_fuse_B_dispatch, _coat_fuse_B_kernel


def pattern(tmp_7, in_4):
    tmp_8 = 0.22941573387056177 * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 3137, 152)
    return (tmp_11,)


def replacement_args(tmp_7, in_4):
    return (tmp_7, in_4, "0.22941573387056177")


def replacement_func():
    return coat_fuse_B_dispatch