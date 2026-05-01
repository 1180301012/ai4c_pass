"""
Pass: FuseDropoutCastLinear_float16
Matches the RECT_L float16 pattern:
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to    = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
Dropout is a no-op (p=0.0, training=False); the .to() cast is folded into
the kernel wrapper via x.to(weight.dtype).
"""
import torch
from pass_dir.shared_linear import fused_linear


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear