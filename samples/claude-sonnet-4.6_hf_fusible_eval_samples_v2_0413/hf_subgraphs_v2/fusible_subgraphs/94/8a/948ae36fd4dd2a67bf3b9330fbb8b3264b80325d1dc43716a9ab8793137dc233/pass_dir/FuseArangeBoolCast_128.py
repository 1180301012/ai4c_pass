"""
Optimization pass: fuse torch.arange(0,128) + in_0.to(bool) for N=128.
Matches graphs with arange end=128 (e.g., input shapes [64,128], [16,128]).
"""
import torch
from torch import device
from pass_dir.arange_bool_cast_kernels import fused_arange_bool_cast


def pattern(in_0):
    tmp_1 = torch.arange(0, 128, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_arange_bool_cast