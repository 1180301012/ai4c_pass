import torch
from pass_dir.repvit_head_shared import repvit_head_dispatch


def pattern(in_4, in_5, in_6):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    return linear


def replacement_args(in_4, in_5, in_6):
    return (in_4, in_5, in_6, "linear")


def replacement_func():
    return repvit_head_dispatch