import torch
import triton
import triton.language as tl
from torch import device

from pass_dir.shared_arange_bool_cast import shared_replacement_func


def pattern(tmp_1, in_0: torch.Tensor):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(tmp_1, in_0: torch.Tensor):
    return (tmp_1, in_0)


def replacement_func():
    return shared_replacement_func()