import torch
import triton
import triton.language as tl

# Match transpose + multiply (working pattern with perfect correctness)
def pattern(tmp_1, in_3):
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    return tmp_3

def replacement_args(tmp_1, in_3):
    return (tmp_1, in_3)

@torch.fx.wrap
def fused_transpose_mul(tmp_1, in_3):
    # Use PyTorch's native operations for best correctness
    return in_3 * tmp_1.transpose(-1, -2)

def replacement_func():
    return fused_transpose_mul