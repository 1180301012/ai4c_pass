import torch
from pass_dir.fused_add_layernorm_shared import fused_add_layernorm_dispatch


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (16,), in_1, in_0, 1e-05)
    tmp_4 = torch.rand([])
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, 'n16')


def replacement_func():
    return fused_add_layernorm_dispatch