import torch
from torch import device as _device


@torch.fx.wrap
def _fused_pos_ids(in_3):
    # [B3, N] int64 -> [B3, 1, N] float32
    return in_3.float().unsqueeze(1)


def pattern(in_3):
    tmp_19 = in_3[slice(None, None, None), None, slice(None, None, None)]
    tmp_20 = tmp_19.float()
    tmp_22 = tmp_20.float()
    return tmp_22


def replacement_args(in_3):
    return (in_3,)


def replacement_func():
    return _fused_pos_ids