import torch
import triton
import triton.language as tl
import math

# Import the shared kernel and wrapper functions
from pass_dir.FuseRollSliceViewAddLayerNorm import fused_kernel_row, fused_kernel_wrapper_133_96

# Pattern matching function - matches the float32 133x133/96 variant
# This pattern mirrors the float32 model.py format exactly (without extra parens on indexing,
# without semicolons, with tmp_0/tmp_1 aliases)
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), tmp_1, tmp_0, 1e-05)
    return (tmp_8, tmp_9)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_kernel_wrapper_133_96