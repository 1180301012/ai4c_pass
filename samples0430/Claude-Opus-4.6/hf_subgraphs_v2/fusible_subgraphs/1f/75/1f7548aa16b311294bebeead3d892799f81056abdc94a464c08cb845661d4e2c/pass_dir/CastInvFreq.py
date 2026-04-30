import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_dispatch import dispatch_fn


def pattern(in_1):
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_21 = tmp_18.float()
    return tmp_21


def replacement_args(in_1):
    return (in_1, in_1, "inv_freq")


def replacement_func():
    return dispatch_fn