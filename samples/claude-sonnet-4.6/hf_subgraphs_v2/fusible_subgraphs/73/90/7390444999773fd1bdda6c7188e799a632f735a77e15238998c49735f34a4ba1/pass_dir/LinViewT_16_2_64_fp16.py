import torch
import triton
import triton.language as tl
from pass_dir.shared_linvt import fused_linear_view_transpose


def pattern(hidden, weight, bias):
    lin = torch.nn.functional.linear(hidden, weight, bias)
    v = lin.view(16, -1, 2, 64)
    v_t = v.transpose(1, 2)
    v_cast = v_t.to(torch.float16)
    return v_cast


def replacement_args(hidden, weight, bias):
    return (hidden, weight, bias, 2, 64)


def replacement_func():
    return fused_linear_view_transpose