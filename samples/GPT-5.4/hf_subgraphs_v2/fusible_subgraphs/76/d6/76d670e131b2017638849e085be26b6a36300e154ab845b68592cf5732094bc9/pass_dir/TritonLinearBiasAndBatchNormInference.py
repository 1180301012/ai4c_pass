import torch
from pass_dir.shared_linear_bn_dispatch import replacement_func


def pattern(in_4, in_5, in_6):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear


def replacement_args(in_4, in_5, in_6):
    return (in_4, in_5, in_6, "linear")