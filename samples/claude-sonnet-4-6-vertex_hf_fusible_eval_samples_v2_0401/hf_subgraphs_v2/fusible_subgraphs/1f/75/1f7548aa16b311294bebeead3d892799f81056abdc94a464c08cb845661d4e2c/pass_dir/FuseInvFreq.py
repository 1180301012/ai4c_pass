import torch
from torch import device as _device


@torch.fx.wrap
def _fused_inv_freq(in_1):
    # [F] any float dtype -> [1, F, 1] float32
    return in_1.float().unsqueeze(0).unsqueeze(-1)


def pattern(in_1):
    tmp_15 = in_1[None, slice(None, None, None), None]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(_device(type='cuda', index=0))
    tmp_21 = tmp_18.float()
    return tmp_21


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return _fused_inv_freq