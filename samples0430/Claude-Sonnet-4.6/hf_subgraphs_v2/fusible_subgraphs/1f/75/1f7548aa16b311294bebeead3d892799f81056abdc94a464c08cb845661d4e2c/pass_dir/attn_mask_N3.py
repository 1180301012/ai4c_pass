"""
Attention mask fused kernel for N=3 (sequence length 3).
Uses shared dispatch_wrapper (route="attn_mask_3") to satisfy replacement_func_limit.
"""

import torch
from torch import device
from pass_dir.dispatch import dispatch_wrapper

_N = 3


def pattern(in_0, in_2):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(_N, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(_N, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = torch.ops.aten.le(tmp_7, tmp_8)
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(in_0, in_2):
    return (in_0, in_2, "attn_mask_3")


def replacement_func():
    return dispatch_wrapper