import torch
from pass_dir.shared_kernel import fused_max_sub_softmax


def pattern(in_1):
    """Matches view(32, 512, -1) found in the bfloat16/7 graph."""
    return in_1.view(32, 512, -1)


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return fused_max_sub_softmax