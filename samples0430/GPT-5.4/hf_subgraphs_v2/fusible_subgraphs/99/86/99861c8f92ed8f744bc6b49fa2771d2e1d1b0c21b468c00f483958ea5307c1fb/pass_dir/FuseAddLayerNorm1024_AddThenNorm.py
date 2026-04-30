import torch
from pass_dir.fused_add_layernorm_common import shared_dispatch


def pattern(in_2, in_3):
    tmp_2 = in_2 + in_3
    return tmp_2


def replacement_args(in_2, in_3):
    return (in_2, in_3, "add")


def replacement_func():
    return shared_dispatch