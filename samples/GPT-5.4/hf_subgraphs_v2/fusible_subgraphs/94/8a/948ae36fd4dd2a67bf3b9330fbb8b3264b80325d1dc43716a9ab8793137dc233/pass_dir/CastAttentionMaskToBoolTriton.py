import torch
from torch import device

from pass_dir.shared_cached_arange_bool_cast import replacement_cast_bool_only


def pattern(in_0):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return replacement_cast_bool_only