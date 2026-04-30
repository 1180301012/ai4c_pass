"""
Pass: FuseLinearDropoutTranspose_return_B

Same as return_A but for dropout(p=0.0, training=False).
"""
import torch
from pass_dir.linear_transpose_kernel import fused_linear_bias


def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    out = torch.nn.functional.dropout(linear, 0.0, False, False)
    return out


def replacement_args(bias, weight, x):
    # Return shape values as FX proxies — the framework traces these and records
    # them as graph nodes.  No int() wrapping.
    return (bias, weight, x, weight.shape[0], x.shape[2], x.shape[0], x.shape[1])


def replacement_func():
    return fused_linear_bias