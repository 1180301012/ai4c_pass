import torch
from pass_dir.shared_relu_add_gap import replacement_func


def pattern(x):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    return tmp_0


def replacement_args(x):
    return (x, "pool_only")