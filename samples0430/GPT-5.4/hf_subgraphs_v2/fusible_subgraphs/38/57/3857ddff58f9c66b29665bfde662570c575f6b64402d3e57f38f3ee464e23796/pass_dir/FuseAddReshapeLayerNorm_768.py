import torch
import triton
import triton.language as tl
from pass_dir.fused_add_layernorm_shared import fused_add_reshape_layernorm


def pattern(in_0, in_1):
    tmp_2 = in_0 + in_1
    tmp_3 = tmp_2.reshape(-1, 768)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_reshape_layernorm