import torch
from pass_dir.shared_ops import fused_dispatch


def pattern(in_3):
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    # Route string appended as last arg for shared dispatch wrapper
    return (in_3, None, None, "sumdiv")


def replacement_func():
    return fused_dispatch