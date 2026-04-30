import torch
import triton
import triton.language as tl
from pass_dir.fused_linear_kernel import dispatch_fused_linear_view_transpose

def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(128, -1, 8, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.float16)
    return to

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3, "128_8_64_cf16")

def replacement_func():
    return dispatch_fused_linear_view_transpose