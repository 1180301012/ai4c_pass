import torch
from pass_dir.shared_dmnet_tail import shared_dmnet_dispatch


def pattern(x, in_0, in_1, in_2, in_3):
    tmp_6 = torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6


def replacement_args(x, in_0, in_1, in_2, in_3):
    return (x, in_0, in_1, in_2, in_3, "bn")


def replacement_func():
    return shared_dmnet_dispatch