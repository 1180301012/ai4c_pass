"""
Pass: FuseGeluAddLayerNorm_256_8_6

Matches the pattern for C=256, H=8, W=6 (pose_hrformer_small_start2123_end2134_97).

Fuses: gelu + flatten(2) + transpose(1,2) + contiguous + add + permute×2 +
       view(1,256,8,6) + view(1,256,-1) + permute(0,2,1) + layer_norm(256,) + view(1,8,6,256)
into a single Triton kernel, eliminating all intermediate tensor allocations.
"""

import torch
from pass_dir.shared_gelu_add_ln import run_fused_gelu_add_layernorm


# ── pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    tmp_2  = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3  = tmp_2.flatten(2)
    tmp_4  = tmp_3.transpose(1, 2)
    tmp_5  = tmp_4.contiguous()
    tmp_6  = in_3 + tmp_5
    tmp_7  = tmp_6.permute(0, 2, 1)
    tmp_8  = tmp_7.view(1, 256, 8, 6)
    tmp_9  = tmp_8.view(1, 256, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 8, 6, 256)
    return (tmp_10, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return run_fused_gelu_add_layernorm