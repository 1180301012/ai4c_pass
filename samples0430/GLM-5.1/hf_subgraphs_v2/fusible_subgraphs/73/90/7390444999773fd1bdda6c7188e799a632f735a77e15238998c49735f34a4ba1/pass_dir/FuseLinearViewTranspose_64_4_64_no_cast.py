import torch
import triton
import triton.language as tl
from pass_dir.fused_linear_kernel import dispatch_fused_linear_view_transpose

def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(64, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3, "64_4_64_nc")

def replacement_func():
    return dispatch_fused_linear_view_transpose