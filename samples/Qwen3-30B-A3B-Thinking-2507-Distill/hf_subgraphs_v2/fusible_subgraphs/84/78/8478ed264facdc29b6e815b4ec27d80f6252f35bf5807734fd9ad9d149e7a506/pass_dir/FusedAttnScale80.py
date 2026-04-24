"""
Pass: Replace permute(0,2,1,3) + contiguous with a fast Triton kernel.
Matches: x.permute(0,2,1,3).contiguous()
Input shape: [B, H, Sq, D] → Output shape: [B, Sq, H*D]
"""

import torch
from pass_dir.shared_fused_attn import dispatch_perm_cont


def pattern(x):
    tmp_5 = x.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(x):
    return (x, "permute")


def replacement_func():
    return dispatch_perm_cont