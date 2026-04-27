import torch
from pass_dir.shared_kernels import dispatch_bn_pool


def pattern(x):
    return torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)


def replacement_args(x):
    # Pad unused BN-param slots with None; append route tag
    return (x, None, None, None, None, "pool2d")


def replacement_func():
    return dispatch_bn_pool