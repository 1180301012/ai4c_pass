import torch
from pass_dir.fused_position_add_layernorm_common import replacement_func


def pattern(in_0, in_2, in_3):
    tmp_14 = torch.nn.functional.layer_norm(in_0, (1024,), in_3, in_2, 1e-05)
    return tmp_14


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)