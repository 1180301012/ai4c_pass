"""
Pass 2: match the spatial mean reduction (after Pass 1 has replaced BN+add+relu).
Mean output (tmp_7) is the only observable output of this sub-graph.
"""
import torch
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x, "mean_only")


def replacement_func():
    return dispatch_wrapper