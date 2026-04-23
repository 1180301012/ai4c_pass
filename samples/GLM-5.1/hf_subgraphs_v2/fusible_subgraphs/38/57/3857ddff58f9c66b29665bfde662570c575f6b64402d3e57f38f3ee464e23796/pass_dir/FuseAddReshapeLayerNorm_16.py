import torch
from pass_dir.layernorm_kernel import triton_layernorm


def pattern(in_0, in_1, tmp_3):
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, tmp_3):
    return (in_0, in_1, tmp_3)


def replacement_func():
    return triton_layernorm