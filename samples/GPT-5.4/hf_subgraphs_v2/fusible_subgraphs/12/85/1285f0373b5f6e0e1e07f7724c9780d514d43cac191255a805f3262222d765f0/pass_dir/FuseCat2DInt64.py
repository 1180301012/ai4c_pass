import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch_ops import dispatch_shared


def pattern(tmp_1, in_1):
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return tmp_9


def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1, 'cat2d_i64')


def replacement_func():
    return dispatch_shared