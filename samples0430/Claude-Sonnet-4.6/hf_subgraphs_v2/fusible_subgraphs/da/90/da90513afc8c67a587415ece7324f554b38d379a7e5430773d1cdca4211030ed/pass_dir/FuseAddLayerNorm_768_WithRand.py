import torch
from pass_dir.fused_add_layernorm_shared import dispatch_fused_add_layernorm


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    tmp_4 = torch.rand([])
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_768_wr")


def replacement_func():
    return dispatch_fused_add_layernorm