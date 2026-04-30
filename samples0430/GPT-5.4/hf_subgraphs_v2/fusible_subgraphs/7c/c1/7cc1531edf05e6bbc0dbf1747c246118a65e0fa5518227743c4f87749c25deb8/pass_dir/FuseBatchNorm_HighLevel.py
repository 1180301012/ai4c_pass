import torch
from pass_dir.shared_fused_pool_bn_silu import shared_replacement_func


def pattern(in_0, in_1, in_2, in_3, x):
    return torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)


def replacement_args(in_0, in_1, in_2, in_3, x):
    return (in_0, in_1, in_2, in_3, x, "bn_only_static")


def replacement_func():
    return shared_replacement_func()