import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(conv2d, in_2):
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(conv2d, in_2):
    return (conv2d, in_2, 'sigmoid_interpolate_mul')