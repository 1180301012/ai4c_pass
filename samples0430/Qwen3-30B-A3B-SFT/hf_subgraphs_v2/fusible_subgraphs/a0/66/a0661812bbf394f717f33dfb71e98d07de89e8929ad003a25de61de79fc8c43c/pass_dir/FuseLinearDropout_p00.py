"""
Pass: FuseLinearDropout_p00
Matches: linear + dropout(p=0.0, training=False) → single tensor output
Covers: bfloat16 graphs (Aniemore), tiny UniSpeechSat graphs
"""
import torch
from pass_dir.gemm_bias_kernel import fused_linear_bias


def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    out = torch.nn.functional.dropout(linear, 0.0, False, False)
    return out


def replacement_args(bias, weight, x):
    return (bias, weight, x)


def replacement_func():
    return fused_linear_bias