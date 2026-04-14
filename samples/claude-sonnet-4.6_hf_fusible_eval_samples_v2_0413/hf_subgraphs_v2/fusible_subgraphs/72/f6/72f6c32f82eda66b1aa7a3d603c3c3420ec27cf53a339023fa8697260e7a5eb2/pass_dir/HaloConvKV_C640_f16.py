"""
Fused pass for float16 Halo Attention KV projection.
Matches: unfold(2,12,8) -> unfold(3,12,8) -> reshape(8,80,4,-1) ->
         permute(0,2,3,1) -> split([16,64]) -> getitem -> transpose

Input: padded tensor [1, 640, 20, 20] float16  (output of pad(conv2d))
K_out: [8, 4, 16, 144]  float16
V_out: [8, 4, 144, 64]  float16
"""

import torch
from pass_dir.halo_shared import halo_dispatch


def pattern(tmp_2):
    tmp_3  = tmp_2.unfold(2, 12, 8)
    tmp_4  = tmp_3.unfold(3, 12, 8)
    tmp_5  = tmp_4.reshape(8, 80, 4, -1)
    tmp_6  = tmp_5.permute(0, 2, 3, 1)
    split  = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8  = split[0]
    tmp_9  = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(tmp_2):
    return (tmp_2, "c640_f16")


def replacement_func():
    return halo_dispatch