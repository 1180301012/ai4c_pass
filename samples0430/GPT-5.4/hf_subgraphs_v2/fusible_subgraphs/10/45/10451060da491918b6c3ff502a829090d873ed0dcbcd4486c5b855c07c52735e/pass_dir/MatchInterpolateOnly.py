import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(tmp_2):
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    return tmp_3


def replacement_args(tmp_2):
    return (tmp_2, 'interpolate_only')