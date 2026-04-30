import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(tmp_2, in_2):
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(tmp_2, in_2):
    return (tmp_2, in_2, 'interpolate_mul')