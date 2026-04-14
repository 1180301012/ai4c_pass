"""
Pass: BeitRelPosIdx24
Matches the BEiT relative position bias index computation for N=24 grid.
Applies to: beit-base-patch16-384 (float16, float32 variants).

Replaces the expensive [2, 576, 576] intermediate-tensor path with a
single Triton kernel computing the flat int64 index table directly.
"""

import torch
from pass_dir.beit_rel_pos_kernels import beit_dispatch


def pattern():
    # Build N×N grid coordinates and compute pairwise differences
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)

    # Broadcast subtraction creates [2, 576, 576] tensor (the bottleneck)
    tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
    tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()

    # In-place transform: row_diff += 23, col_diff += 23, row_diff *= 47
    tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_13 += 23
    tmp_14 = tmp_13
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_14

    tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_16 += 23
    tmp_17 = tmp_16
    tmp_12[slice(None, None, None), slice(None, None, None), 1] = tmp_17

    tmp_19 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_19 *= 47
    tmp_20 = tmp_19
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_20

    # Build output table: zeros (577,577), fill inner, set CLS sentinels
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
    tmp_22[0, slice(0, None, None)] = 2209
    tmp_22[slice(0, None, None), 0] = 2210
    tmp_22[0, 0] = 2211
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args():
    return ("24",)


def replacement_func():
    return beit_dispatch