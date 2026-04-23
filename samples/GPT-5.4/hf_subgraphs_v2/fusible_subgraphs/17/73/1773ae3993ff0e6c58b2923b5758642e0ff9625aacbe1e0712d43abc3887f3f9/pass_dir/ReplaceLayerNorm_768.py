import torch
import triton
import triton.language as tl

from pass_dir.yolos_layernorm_shared import (
    yolos_layer_norm_replacement_args,
    yolos_layer_norm_replacement_func,
)


def pattern(in_4, in_1, in_0):
    tmp_3 = torch.nn.functional.layer_norm(in_4, (768,), in_1, in_0, 1e-12)
    return tmp_3


def replacement_args(in_4, in_1, in_0):
    return yolos_layer_norm_replacement_args(in_4, in_1, in_0)


def replacement_func():
    return yolos_layer_norm_replacement_func()