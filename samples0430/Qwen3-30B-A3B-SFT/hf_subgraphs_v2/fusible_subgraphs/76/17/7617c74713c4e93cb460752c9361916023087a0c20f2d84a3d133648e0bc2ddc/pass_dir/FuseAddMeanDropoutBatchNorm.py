import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(in_4, in_5):
    """
    Matches add + spatial mean. Returns the mean result [N, C].
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    # Route "add_mean": in_4=in4, in_5=in5; c,d,e=None unused
    return (in_4, in_5, None, None, None, "add_mean")


def replacement_func():
    return dispatch_wrapper