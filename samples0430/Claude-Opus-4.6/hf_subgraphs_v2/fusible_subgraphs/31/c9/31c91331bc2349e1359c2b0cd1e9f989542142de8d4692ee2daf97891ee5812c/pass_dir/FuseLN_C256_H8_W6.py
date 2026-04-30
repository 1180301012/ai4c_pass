import torch
from pass_dir.shared_dispatch import fused_dispatch


def pattern(in_0, in_1, tmp_10):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 8, 6, 256)
    return tmp_12


def replacement_args(in_0, in_1, tmp_10):
    return (in_0, in_1, tmp_10, 8, 6)


def replacement_func():
    return fused_dispatch