"""
Pass for Scaled Dot-Product Attention with scale = 6.928203230275509 (≈ sqrt(50))
and dropout p = 0.1.

Covers: hf-tiny-model-private-MobileViT
  - start152_end160_12  (float32,  x4)
  - start93_end101_6    (float16, x2)
"""
import torch
import triton
import triton.language as tl
from pass_dir.flash_attn_shared import flash_attn_wrapper

_SCALE_6928 = 6.928203230275509


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 6.928203230275509
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, _SCALE_6928, 1.0 / _SCALE_6928)


def replacement_func():
    return flash_attn_wrapper