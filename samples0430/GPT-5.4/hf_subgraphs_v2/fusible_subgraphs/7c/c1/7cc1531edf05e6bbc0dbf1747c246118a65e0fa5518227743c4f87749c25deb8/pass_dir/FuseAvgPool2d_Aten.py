import torch
from pass_dir.shared_fused_pool_bn_silu import shared_replacement_func


def pattern(x):
    return torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2], [0, 0], False, True, None)


def replacement_args(x):
    return (x, "avgpool_only")


def replacement_func():
    return shared_replacement_func()