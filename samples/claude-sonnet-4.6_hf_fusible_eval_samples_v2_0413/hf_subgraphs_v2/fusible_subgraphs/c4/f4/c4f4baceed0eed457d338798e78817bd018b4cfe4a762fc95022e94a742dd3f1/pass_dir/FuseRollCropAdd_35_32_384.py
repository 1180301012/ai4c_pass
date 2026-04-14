"""
Pass: Fuse view+roll+slice+contiguous+view+add for H=35, CROP=32, C=384.
Pattern returns a SINGLE tensor (tmp_8), avoiding multi-output tuple issues.
Matches all dtype variants (float16, bfloat16, float32).
"""
import torch
from pass_dir.triton_fused_dispatch import fused_dispatch


def pattern(in_2, in_3):
    tmp_3 = in_3.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    return in_2 + tmp_7


def replacement_args(in_2, in_3):
    return (in_2, in_3, in_2, "add_384")


def replacement_func():
    return fused_dispatch