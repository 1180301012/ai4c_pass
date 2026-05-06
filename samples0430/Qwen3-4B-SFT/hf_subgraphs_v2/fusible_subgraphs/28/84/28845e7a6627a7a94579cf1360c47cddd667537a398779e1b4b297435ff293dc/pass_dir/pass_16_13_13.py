"""
Pass: fuse add in_1 + in_0
_ACTIVE for all 5 graphs (add is shape-independent).
"""
import torch
from pass_dir.shared_kernel import dispatch_fused_add_mask_softmax


def pattern(in_0, in_1):
    return in_1 + in_0


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_13_13")


def replacement_func():
    return dispatch_fused_add_mask_softmax