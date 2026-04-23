import torch
from pass_dir.shared_fused_add_layernorm import fused_add_layernorm_dispatch


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_2, in_3, in_1, in_0, "tmp4_tmp2")


def replacement_func():
    return fused_add_layernorm_dispatch