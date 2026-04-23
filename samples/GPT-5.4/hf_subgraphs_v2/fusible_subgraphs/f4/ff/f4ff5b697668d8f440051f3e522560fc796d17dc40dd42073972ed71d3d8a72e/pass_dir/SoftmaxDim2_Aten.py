import torch
from pass_dir.common_softmax_expect import replacement_func


def pattern(in_2):
    return torch.ops.aten.softmax.int(in_2, 2)


def replacement_args(in_2):
    return (in_2,)