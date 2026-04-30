"""
Pass: FuseLinearDropout_p01
Matches: linear + dropout(p=0.1, training=False) → single tensor output
Covers: float16 graphs (Rajaram1996, distilhubert)
"""
import torch
from pass_dir.gemm_bias_kernel import fused_linear_bias


def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    out = torch.nn.functional.dropout(linear, 0.1, False, False)
    return out


def replacement_args(bias, weight, x):
    return (bias, weight, x)


def replacement_func():
    return fused_linear_bias