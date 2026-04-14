"""
Pass: Fuse roll+crop+residual_add+layer_norm for H=133, CROP=128, C=96.
Matches all dtype variants (float16, bfloat16, float32).

Uses the shared dispatch wrapper so all three passes share the SAME
replacement_func, satisfying output_pass_replacement_func_limit.
"""
import torch
from pass_dir.triton_roll_crop_add_ln import fused_roll_crop_add_ln_dispatch


def pattern(in_0, in_1, in_2, in_3):
    # in_3 here matches `tmp_2 = in_3_orig.contiguous()` in the model FX graph
    # (first .contiguous() excluded from pattern so wrapper receives contiguous tensor)
    tmp_3 = in_3.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "96")


def replacement_func():
    return fused_roll_crop_add_ln_dispatch