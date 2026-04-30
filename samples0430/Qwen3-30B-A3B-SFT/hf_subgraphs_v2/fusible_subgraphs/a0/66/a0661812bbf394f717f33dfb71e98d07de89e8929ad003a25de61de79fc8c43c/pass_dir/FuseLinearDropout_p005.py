"""
Pass: FuseLinearDropout_p005
Matches: linear + dropout(p=0.05, training=False) → single tensor output
Covers: bfloat16 graphs (Aniemore) with p=0.05.
"""
import torch
from pass_dir.linear_transpose_kernel import fused_linear_bias


def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    out = torch.nn.functional.dropout(linear, 0.05, False, False)
    return out


def replacement_args(bias, weight, x):
    # Pass shape values as FX proxies so the framework can trace them
    return (bias, weight, x, weight.shape[0], x.shape[2], x.shape[0], x.shape[1])


def replacement_func():
    return fused_linear_bias