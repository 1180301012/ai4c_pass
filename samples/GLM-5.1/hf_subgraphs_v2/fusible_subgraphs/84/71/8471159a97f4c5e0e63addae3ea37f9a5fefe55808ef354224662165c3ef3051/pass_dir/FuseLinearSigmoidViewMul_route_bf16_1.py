import torch
import triton
import triton.language as tl
from pass_dir.se_kernel import fused_se_dispatch

# Pattern matching function - must match model.py EXACTLY (no cleanup statements)
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return (tmp_5,)

# Argument extraction - include route string as last arg
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "se_block_view_1_64_1_1")

def replacement_func():
    return fused_se_dispatch