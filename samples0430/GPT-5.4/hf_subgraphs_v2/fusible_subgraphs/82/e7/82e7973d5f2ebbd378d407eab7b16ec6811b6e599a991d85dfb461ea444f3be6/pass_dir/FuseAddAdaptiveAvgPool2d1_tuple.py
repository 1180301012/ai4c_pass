import torch
from pass_dir.shared_relu_add_gap import replacement_func


def pattern(x, y):
    tmp_0 = x + y
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(x, y):
    return (x, y, "add_pool")