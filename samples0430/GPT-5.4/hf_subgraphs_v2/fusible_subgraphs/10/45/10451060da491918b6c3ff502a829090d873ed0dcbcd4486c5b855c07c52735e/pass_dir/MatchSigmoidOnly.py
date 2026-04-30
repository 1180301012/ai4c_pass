import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(conv2d):
    tmp_2 = torch.sigmoid(conv2d)
    return tmp_2


def replacement_args(conv2d):
    return (conv2d, 'sigmoid_only')