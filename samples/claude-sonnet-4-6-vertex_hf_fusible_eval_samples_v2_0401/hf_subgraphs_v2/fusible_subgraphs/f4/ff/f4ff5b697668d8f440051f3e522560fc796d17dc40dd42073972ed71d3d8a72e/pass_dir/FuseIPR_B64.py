"""
Fused IPR pass for batch size B=64.
Matches: softmax -> reshape(-1,17,64,64) -> mul(lx) -> reshape(64,17,-1) -> sum
                                          -> mul(ly) -> reshape(64,17,-1) -> sum -> cat
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from ipr_kernel_impl import ipr_softmax_weighted_sum_kernel


@torch.fx.wrap
def ipr_fused_B64(in_0, in_1, in_2):
    B, K, H, W = 64, 17, 64, 64
    out_heatmap = in_2.new_empty(B, K, H, W)
    out_coords  = in_2.new_empty(B, K, 2)
    in0_flat = in_0.reshape(-1)
    in1_flat = in_1.reshape(-1)
    ipr_softmax_weighted_sum_kernel[(B * K,)](
        in_2, in0_flat, in1_flat,
        out_heatmap, out_coords,
        HW=H * W, W=W,
        BLOCK_SIZE=H * W,
        num_warps=16,
    )
    return out_heatmap, out_coords


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(64, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(64, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return ipr_fused_B64