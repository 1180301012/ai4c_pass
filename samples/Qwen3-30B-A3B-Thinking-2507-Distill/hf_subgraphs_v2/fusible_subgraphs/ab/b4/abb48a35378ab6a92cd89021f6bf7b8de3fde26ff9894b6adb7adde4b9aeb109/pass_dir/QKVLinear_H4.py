"""Pass: QKVLinear_H4 — single-output: linear+reshape(3,4,48)+permute."""
import torch
from pass_dir.qkv_fused_kernel import _kv_linear_permute


def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = linear.reshape(1, 197, 3, 4, 48)
    tmp_3  = tmp_2.permute(2, 0, 3, 1, 4)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _kv_linear_permute