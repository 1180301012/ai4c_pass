"""
Optimization pass for the broadcast-multiply pattern:
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2

Single-output pattern (no tuple return) — FX-compatible replacement.
The RECT_L graph (float16 variant) uses N=256, D=128.
Uses shared dispatch wrapper from kernels.py.
"""
import torch
from pass_dir.kernels import dispatch_fused_view_mul


def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2, "256_128")


def replacement_func():
    return dispatch_fused_view_mul