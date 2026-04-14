"""
Pass: Fuse view+roll+slice+contiguous+view+add for H=133, CROP=128, C=96.
Pattern returns a SINGLE tensor (tmp_8), avoiding multi-output tuple issues.
Matches all dtype variants (float16, bfloat16, float32).
in_3 in the pattern corresponds to tmp_2 (already contiguous) in the FX graph.
"""
import torch
from pass_dir.triton_fused_dispatch import fused_dispatch


def pattern(in_2, in_3):
    tmp_3 = in_3.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    return in_2 + tmp_7


def replacement_args(in_2, in_3):
    return (in_2, in_3, in_2, "add_96")


def replacement_func():
    return fused_dispatch