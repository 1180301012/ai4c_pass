import torch
import operator
from torch import device
from pass_dir.shared_attn_kernel import _run_attn_mask_n2

torch.fx.wrap(operator.le)


def pattern(in_0, in_2):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(2, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_5 = tmp_2[(slice(None, None, None), tmp_3)]
    tmp_7 = torch.arange(2, device=device(type='cuda', index=0))
    tmp_7 += 0
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = operator.le(tmp_7, tmp_8)
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@torch.fx.wrap
def fused_attn_mask_2(in_0, in_2):
    return _run_attn_mask_n2(in_0, in_2)


def replacement_func():
    return fused_attn_mask_2