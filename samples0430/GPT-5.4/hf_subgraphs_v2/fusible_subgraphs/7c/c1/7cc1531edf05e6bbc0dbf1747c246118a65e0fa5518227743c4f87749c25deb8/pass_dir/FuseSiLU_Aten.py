import torch
from pass_dir.shared_fused_pool_bn_silu import shared_replacement_func


def pattern(x):
    return torch.ops.aten.silu.default(x)


def replacement_args(x):
    return (x, "silu_only")


def replacement_func():
    return shared_replacement_func()