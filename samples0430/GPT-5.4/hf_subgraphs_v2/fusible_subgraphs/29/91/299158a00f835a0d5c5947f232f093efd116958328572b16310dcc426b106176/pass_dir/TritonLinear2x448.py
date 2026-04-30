import torch
import triton
import triton.language as tl
from pass_dir.classifier_head_shared import classifier_head_dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "linear")


def replacement_func():
    return classifier_head_dispatch