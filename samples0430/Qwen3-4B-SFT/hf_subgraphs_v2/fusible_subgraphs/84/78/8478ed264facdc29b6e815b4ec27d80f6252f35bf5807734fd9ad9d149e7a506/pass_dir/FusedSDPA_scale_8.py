"""
Pass for Scaled Dot-Product Attention with scale = 8.0 (≈ sqrt(64)), dropout p=0.0.
Matches the matmul→scale→softmax→dropout(0)→matmul→permute→contiguous subgraph
and returns the contiguous [B,H,S,D] tensor.  The downstream view() from the
original graph is a zero-cost reshape and is left untouched.

Covers: face-parsing_start149 (bf16/f16), face-parsing_start21 (f16/f32/f16),
        apple_mobilevit-small, nvidia_mit-b0_start279 (f16/bf16/f32),
        nvidia_mit-b0_start197 (bf32/bf16/f16), nvidia_mit-b0_start109 (f16/bf16/f32).
"""
import torch
import triton
import triton.language as tl
import operator
from pass_dir.flash_attn_shared import (
    _flash_attn_fused_kernel,
    flash_attn_wrapper,
)

_SCALE_80 = 8.0
_scale_arg = 8.0   # placeholder value passed to replacement_args


def pattern(in_0, in_1, in_2):
    matmul    = torch.matmul(in_0, in_1)
    tmp_1     = matmul / 8.0
    tmp_2     = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3     = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    matmul_1  = torch.matmul(tmp_3, in_2)
    tmp_5     = matmul_1.permute(0, 2, 1, 3)
    tmp_6     = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, _SCALE_80, 1.0 / _SCALE_80)


def replacement_func():
    return flash_attn_wrapper