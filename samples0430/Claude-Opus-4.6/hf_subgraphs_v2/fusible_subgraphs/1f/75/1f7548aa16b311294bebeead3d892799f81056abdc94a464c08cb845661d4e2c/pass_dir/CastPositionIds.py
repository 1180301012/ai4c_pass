import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_dispatch import dispatch_fn


def pattern(in_3):
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_22 = tmp_20.float()
    return tmp_22


def replacement_args(in_3):
    return (in_3, in_3, "pos_ids")


def replacement_func():
    return dispatch_fn