import torch
import triton
import triton.language as tl
from pass_dir.shared_add_layernorm import shared_dispatch


def pattern(in_0, in_1, tmp_3):
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, tmp_3):
    return (tmp_3, in_1, in_0, "layer_norm_16")


def replacement_func():
    return shared_dispatch