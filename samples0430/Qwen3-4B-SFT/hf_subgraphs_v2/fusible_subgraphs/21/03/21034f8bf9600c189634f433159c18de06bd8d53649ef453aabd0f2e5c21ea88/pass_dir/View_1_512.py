import torch
from pass_dir.shared_kernel import fused_max_sub_softmax


def pattern(in_1):
    """Matches view(1, 512, -1) found in the float32/0 and float16/9 graphs."""
    return in_1.view(1, 512, -1)


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return fused_max_sub_softmax