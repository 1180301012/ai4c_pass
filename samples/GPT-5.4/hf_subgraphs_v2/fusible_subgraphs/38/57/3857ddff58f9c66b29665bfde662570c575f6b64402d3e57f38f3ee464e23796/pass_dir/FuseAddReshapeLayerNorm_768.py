import torch
import triton
import triton.language as tl
from pass_dir.shared_add_layernorm import shared_dispatch


def pattern(in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 768)
    return tmp_3


def replacement_args(in_2, in_3):
    return (in_2, in_3, "add_reshape_768")


def replacement_func():
    return shared_dispatch