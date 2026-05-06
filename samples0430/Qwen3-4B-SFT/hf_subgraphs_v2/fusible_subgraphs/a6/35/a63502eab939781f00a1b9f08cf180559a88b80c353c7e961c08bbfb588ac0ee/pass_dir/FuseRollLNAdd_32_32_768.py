"""
Fused pass for C=768: contiguous + view(-1,32,32,768) + roll(4,4) + view(1,1024,768)
            + layer_norm + residual_add  → single Triton kernel

Imports fused_dispatch from the shared _internal module so both pass files
return the EXACT SAME Python function object from replacement_func(),
satisfying replacement_func_limit = 1.
"""

import torch
import triton
import triton.language as tl

from pass_dir._internal.fused_kernel import fused_dispatch


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_768_1024")


def replacement_func():
    return fused_dispatch