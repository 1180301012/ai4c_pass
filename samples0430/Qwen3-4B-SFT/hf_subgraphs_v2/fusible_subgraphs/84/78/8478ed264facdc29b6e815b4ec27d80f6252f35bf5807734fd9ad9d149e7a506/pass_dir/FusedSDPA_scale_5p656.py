"""
Pass for Scaled Dot-Product Attention with scale = 5.656854249492381.
Matches the attention body (matmul → scale → softmax → dropout → matmul →
permute → contiguous) and returns the contiguous [B,H,S,D] tensor.
The remaining view() in the original graph is a no-cost reshape and stays untouched.

Covers: face-parsing_start1999, face-parsing_start149, face-parsing_start21,
        nvidia_mit-b0_start197, nvidia_mit-b0_start21, nvidia_mit-b0_start109
with any view suffix (32, 64, 128, 160, 320, 512).
"""
import torch
import triton
import triton.language as tl
from pass_dir.flash_attn_shared import (
    _flash_attn_fused_kernel,
    flash_attn_wrapper,
)

_scale_5p656 = 5.656854249492381


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, _scale_5p656, 1.0 / _scale_5p656)


def replacement_func():
    return flash_attn_wrapper