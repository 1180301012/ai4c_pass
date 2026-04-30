import torch
from pass_dir.shared_dispatch import fused_dispatch


def pattern(in_0, in_1, tmp_10):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (128,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 16, 12, 128)
    return tmp_12


def replacement_args(in_0, in_1, tmp_10):
    return (in_0, in_1, tmp_10, 16, 12)


def replacement_func():
    return fused_dispatch