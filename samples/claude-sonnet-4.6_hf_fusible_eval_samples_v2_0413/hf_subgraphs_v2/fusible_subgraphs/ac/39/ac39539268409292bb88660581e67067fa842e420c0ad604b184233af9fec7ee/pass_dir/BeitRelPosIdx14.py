"""
Pass: BeitRelPosIdx14
Matches the BEiT relative position bias index computation for N=14 grid.
Applies to: BEiT-FaceMask-Finetuned (bfloat16 variant).

Replaces the [2, 196, 196] intermediate-tensor path with a single Triton
kernel computing the flat int64 index table directly.
"""

import torch
from pass_dir.beit_rel_pos_kernels import beit_dispatch


def pattern():
    # Build N×N grid coordinates and compute pairwise differences
    tmp_1 = torch.arange(14)
    tmp_2 = torch.arange(14)
    tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)

    # Broadcast subtraction creates [2, 196, 196] tensor (the bottleneck)
    tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
    tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()

    # In-place transform: row_diff += 13, col_diff += 13, row_diff *= 27
    tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_13 += 13
    tmp_14 = tmp_13
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_14

    tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_16 += 13
    tmp_17 = tmp_16
    tmp_12[slice(None, None, None), slice(None, None, None), 1] = tmp_17

    tmp_19 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_19 *= 27
    tmp_20 = tmp_19
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_20

    # Build output table: zeros (197,197), fill inner, set CLS sentinels
    tmp_22 = torch.zeros(size=(197, 197), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
    tmp_22[0, slice(0, None, None)] = 729
    tmp_22[slice(0, None, None), 0] = 730
    tmp_22[0, 0] = 731
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args():
    return ("14",)


def replacement_func():
    return beit_dispatch