import torch
import triton
import triton.language as tl
from pass_dir.classifier_head_shared import classifier_head_dispatch


def pattern(in_3):
    tmp_3 = in_3.mean(-2)
    return tmp_3


def replacement_args(in_3):
    return (in_3, "mean")


def replacement_func():
    return classifier_head_dispatch