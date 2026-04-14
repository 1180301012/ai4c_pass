"""
Pass: BeitRelPosIdx32
Matches the BEiT relative position bias index computation for N=32 grid.
Applies to: beit-large-patch16-512 (bfloat16, float16, float32 variants).

The pattern matches the entire deterministic index computation subgraph
(no external tensor inputs - uses only constant arange/zeros operations)
and replaces it with a single Triton kernel that computes the flat int64
index table directly, bypassing the expensive [2, 1024, 1024] intermediate.
"""

import torch
from pass_dir.beit_rel_pos_kernels import beit_dispatch


def pattern():
    # Build N×N grid coordinates and compute pairwise differences
    tmp_1 = torch.arange(32)
    tmp_2 = torch.arange(32)
    tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)

    # Broadcast subtraction creates [2, 1024, 1024] tensor (the bottleneck)
    tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
    tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()

    # In-place transform: row_diff += 31, col_diff += 31, row_diff *= 63
    tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_13 += 31
    tmp_14 = tmp_13
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_14

    tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_16 += 31
    tmp_17 = tmp_16
    tmp_12[slice(None, None, None), slice(None, None, None), 1] = tmp_17

    tmp_19 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_19 *= 63
    tmp_20 = tmp_19
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_20

    # Build output table: zeros (1025,1025), fill inner, set CLS sentinels
    tmp_22 = torch.zeros(size=(1025, 1025), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
    tmp_22[0, slice(0, None, None)] = 3969
    tmp_22[slice(0, None, None), 0] = 3970
    tmp_22[0, 0] = 3971
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args():
    return ("32",)


def replacement_func():
    return beit_dispatch