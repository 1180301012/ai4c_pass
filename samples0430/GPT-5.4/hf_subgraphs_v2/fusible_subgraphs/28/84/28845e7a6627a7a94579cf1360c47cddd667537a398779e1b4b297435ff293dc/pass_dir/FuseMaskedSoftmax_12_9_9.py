import torch
from torch import device

from pass_dir.fused_masked_softmax_shared import shared_replacement_func


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(12, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return (tmp_5,)


def replacement_args(in_0, in_1):
    return (in_1, in_0)


def replacement_func():
    return shared_replacement_func()