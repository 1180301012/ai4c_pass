import torch
from torch import device
from pass_dir.shared_mask_prepare import get_replacement_func


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    _set_grad_enabled = torch.set_grad_enabled(False)
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return get_replacement_func()